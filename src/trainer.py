import argparse
import shutil
import time
import yaml

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset
from loss import get_loss
from models import get_model
from models.tools import count_parameters, safe_model_state_dict
from optimizers import get_optimizer
from schedulers import get_scheduler
from tester import Tester
from utils import use_seed, coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from utils.constant import MODEL_FILE
from utils.path import CONFIGS_PATH, MODELS_PATH
from utils.metrics import AverageMeter, RunningMetrics
from utils.logger import get_logger, print_info


PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}]: train_loss = {:.4f}, time/img = {:.4f}s"
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}]: val_loss = {:.4f}"
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}]: LR updated, lr = {}"

TRAIN_METRICS_FILE = "train_metrics.tsv"
VAL_METRICS_FILE = "val_metrics.tsv"


class Trainer:
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    @use_seed()
    def __init__(self, config_path, run_dir):
        self.config_path = coerce_to_path_and_check_exist(config_path)
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(self.run_dir, name="trainer")
        self.print_and_log_info("Trainer initialisation: run directory is {}".format(run_dir))

        shutil.copy(self.config_path, self.run_dir)
        self.print_and_log_info("Config {} copied to run directory".format(self.config_path))

        with open(self.config_path) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        if torch.cuda.is_available():
            type_device = "cuda"
            nb_device = torch.cuda.device_count()
            # XXX: set to False when input image sizes are not fixed
            torch.backends.cudnn.benchmark = cfg["training"].get("cudnn_benchmark", True)

        else:
            type_device = "cpu"
            nb_device = None
        self.device = torch.device(type_device)
        self.print_and_log_info("Using {} device, nb_device is {}".format(type_device, nb_device))

        # Datasets and dataloaders
        self.dataset_kwargs = cfg["dataset"]
        self.dataset_name = self.dataset_kwargs.pop("name")
        train_dataset = get_dataset(self.dataset_name)("train", **self.dataset_kwargs)
        val_dataset = get_dataset(self.dataset_name)("val", **self.dataset_kwargs)
        self.restricted_labels = sorted(self.dataset_kwargs["restricted_labels"])
        self.n_classes = len(self.restricted_labels) + 1
        self.is_val_empty = len(val_dataset) == 0
        self.print_and_log_info("Dataset {} instantiated with {}".format(self.dataset_name, self.dataset_kwargs))
        self.print_and_log_info("Found {} classes, {} train samples, {} val samples"
                                .format(self.n_classes, len(train_dataset), len(val_dataset)))

        self.batch_size = cfg["training"]["batch_size"]
        self.n_workers = cfg["training"]["n_workers"]
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       num_workers=self.n_workers, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        self.print_and_log_info("Dataloaders instantiated with batch_size={} and n_workers={}"
                                .format(self.batch_size, self.n_workers))

        self.n_batches = len(self.train_loader)
        self.n_iterations, self.n_epoches = cfg["training"].get("n_iterations"), cfg["training"].get("n_epoches")
        assert not (self.n_iterations is not None and self.n_epoches is not None)
        if self.n_iterations is not None:
            self.n_epoches = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epoches * len(self.train_loader)

        # Model
        self.model_kwargs = cfg["model"]
        self.model_name = self.model_kwargs.pop("name")
        model = get_model(self.model_name)(self.n_classes, **self.model_kwargs).to(self.device)
        self.model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        self.print_and_log_info("Using model {} with kwargs {}".format(self.model_name, self.model_kwargs))
        self.print_and_log_info('Number of trainable parameters: {}'.format(f'{count_parameters(self.model):,}'))

        # Optimizer
        optimizer_params = cfg["training"]["optimizer"] or {}
        optimizer_name = optimizer_params.pop("name", None)
        self.optimizer = get_optimizer(optimizer_name)(model.parameters(), **optimizer_params)
        self.print_and_log_info("Using optimizer {} with kwargs {}".format(optimizer_name, optimizer_params))

        # Scheduler
        scheduler_params = cfg["training"].get("scheduler", {}) or {}
        scheduler_name = scheduler_params.pop("name", None)
        self.scheduler_update_range = scheduler_params.pop("update_range", "epoch")
        assert self.scheduler_update_range in ["epoch", "batch"]
        if scheduler_name == "multi_step" and isinstance(scheduler_params["milestones"][0], float):
            n_tot = self.n_epoches if self.scheduler_update_range == "epoch" else self.n_iterations
            scheduler_params["milestones"] = [round(m * n_tot) for m in scheduler_params["milestones"]]
        self.scheduler = get_scheduler(scheduler_name)(self.optimizer, **scheduler_params)
        self.cur_lr = -1
        self.print_and_log_info("Using scheduler {} with parameters {}".format(scheduler_name, scheduler_params))

        # Loss
        loss_name = cfg["training"]["loss"]
        self.criterion = get_loss(loss_name)()
        self.print_and_log_info("Using loss {}".format(self.criterion))

        # Pretrained / Resume
        checkpoint_path = cfg["training"].get("pretrained")
        checkpoint_path_resume = cfg["training"].get("resume")
        assert not (checkpoint_path is not None and checkpoint_path_resume is not None)
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1

        # Train metrics
        train_iter_interval = cfg["training"].get("train_stat_interval", self.n_epoches * self.n_batches // 200)
        self.train_stat_interval = train_iter_interval
        self.train_time = AverageMeter()
        self.train_loss = AverageMeter()
        self.train_metrics_path = self.run_dir / TRAIN_METRICS_FILE
        with open(self.train_metrics_path, mode="w") as f:
            f.write("iteration\tepoch\tbatch\ttrain_loss\ttrain_time_per_img\n")

        # Val metrics
        val_iter_interval = cfg["training"].get("val_stat_interval", self.n_epoches * self.n_batches // 100)
        self.val_stat_interval = val_iter_interval
        self.val_loss = AverageMeter()
        self.val_metrics = RunningMetrics(self.restricted_labels)
        self.val_current_score = None
        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        with open(self.val_metrics_path, mode="w") as f:
            f.write("iteration\tepoch\tbatch\tval_loss\t" + "\t".join(self.val_metrics.names) + "\n")

    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info("Loading model from run {}".format(tag))
        path = coerce_to_path_and_check_exist(MODELS_PATH / tag / MODEL_FILE)
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint["model_state"])
            self.model.module.load_state_dict(state)
        self.start_epoch, self.start_batch = 1, 1
        if resume:
            self.start_epoch, self.start_batch = checkpoint["epoch"], checkpoint.get("batch", 0) + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_lr()
        self.print_and_log_info("Checkpoint loaded at epoch {}, batch {}".format(self.start_epoch, self.start_batch-1))

    def _create_external_val_loader_and_monitor(self, dataset_name):
        val_dataset = get_dataset(dataset_name)(split="val", **self.dataset_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        self.print_and_log_info("External {} validation dataset instantiated with kwargs {}: {} samples"
                                .format(dataset_name, self.dataset_kwargs, len(val_dataset)))
        monitor = {}
        monitor["name"] = dataset_name
        monitor["loss"] = AverageMeter()
        monitor["metrics"] = RunningMetrics(val_dataset.restricted_labels, val_dataset.metric_labels)
        monitor["metrics_path"] = self.run_dir / "{}_metrics.tsv".format(dataset_name)
        with open(monitor["metrics_path"], mode="w") as f:
            f.write("iteration\tepoch\tbatch\t{}_loss\t".format(dataset_name) +
                    "\t".join(monitor["metrics"].names) + "\n")

        return val_loader, monitor

    @property
    def score_name(self):
        return self.val_metrics.score_name

    def print_memory_usage(self, prefix):
        usage = {}
        for attr in ["memory_allocated", "max_memory_allocated", "memory_cached", "max_memory_cached"]:
            usage[attr] = getattr(torch.cuda, attr)() * 0.000001
        self.print_and_log_info("{}:\t{}".format(
            prefix, " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()])))

    @use_seed()
    def run(self):
        self.model.train()
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        for epoch in range(self.start_epoch, self.n_epoches + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            if self.scheduler_update_range == "epoch":
                if batch_start == 1:
                    self.update_scheduler(epoch, batch=batch_start)

            for batch, (images, labels) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                if self.scheduler_update_range == "batch":
                    self.update_scheduler(epoch, batch=batch)

                self.single_train_batch_run(images, labels)
                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    self.log_train_metrics(cur_iter, epoch, batch)

                if (cur_iter - prev_val_stat_iter) >= self.val_stat_interval:
                    prev_val_stat_iter = cur_iter
                    self.run_val()
                    self.log_val_metrics(cur_iter, epoch, batch)
                    self.save(epoch=epoch, batch=batch)

        self.print_and_log_info("Training run is over")

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        lr = self.scheduler.get_lr()
        if lr != self.cur_lr:
            self.cur_lr = lr
            msg = PRINT_LR_UPD_FMT.format(epoch, self.n_epoches, batch, self.n_batches, lr)
            self.print_and_log_info(msg)

    def single_train_batch_run(self, images, labels):
        start_time = time.time()
        images, labels = images.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(images), labels)
        loss.backward()
        self.optimizer.step()

        self.train_loss.update(loss.item())
        self.train_time.update((time.time() - start_time) / self.batch_size)

    def log_train_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_TRAIN_STAT_FMT.format(epoch, self.n_epoches, batch, self.n_batches,
                                           self.train_loss.avg, self.train_time.avg)
        self.print_and_log_info(stat)

        with open(self.train_metrics_path, mode="a") as f:
            f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\n"
                    .format(cur_iter, epoch, batch, self.train_loss.avg, self.train_time.avg))

        self.train_loss.reset()
        self.train_time.reset()

    def run_val(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                pred = outputs.data.max(1)[1].cpu().numpy()
                if images.size() == labels.size():
                    gt = labels.data.max(1)[1].cpu().numpy()
                else:
                    gt = labels.cpu().numpy()

                self.val_metrics.update(gt, pred)
                self.val_loss.update(loss.item())

        self.model.train()

    def log_val_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_VAL_STAT_FMT.format(epoch, self.n_epoches, batch, self.n_batches, self.val_loss.avg)
        self.print_and_log_info(stat)

        metrics = self.val_metrics.get()
        self.print_and_log_info("Val metrics: " + ", ".join(["{} = {:.4f}".format(k, v) for k, v in metrics.items()]))

        with open(self.val_metrics_path, mode="a") as f:
            f.write("{}\t{}\t{}\t{:.4f}\t".format(cur_iter, epoch, batch, self.val_loss.avg) +
                    "\t".join(map("{:.4f}".format, metrics.values())) + "\n")

        self.val_current_score = metrics[self.score_name]
        self.val_loss.reset()
        self.val_metrics.reset()

    def save(self, epoch, batch):
        state = {
            "epoch": epoch,
            "batch": batch,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "model_state": self.model.state_dict(),
            "n_classes": self.n_classes,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "score": self.val_current_score,
            "train_resolution": self.dataset_kwargs["img_size"],
            "restricted_labels": self.dataset_kwargs["restricted_labels"],
            "normalize": self.dataset_kwargs["normalize"],
        }

        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info("Model saved at {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, help="Model tag of the experiment", required=True)
    parser.add_argument("-c", "--config", nargs="?", type=str, default="syndoc.yml", help="Config file name")
    parser.add_argument("-s", "--seed", nargs="?", type=int, default=4321, help="Seed number")
    parser.add_argument('-wt', '--with_test', action='store_true', help='Whether to run corresponding Tester')
    args = parser.parse_args()

    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    run_dir = MODELS_PATH / args.tag

    trainer = Trainer(config, run_dir, seed=args.seed)
    trainer.run(seed=args.seed)

    if args.with_test:
        dataset_name = trainer.dataset_name
        output_dir = run_dir / "test_{}".format(dataset_name)

        tester = Tester(output_dir, run_dir / MODEL_FILE, dataset_name, trainer.dataset_kwargs)
        tester.run()
