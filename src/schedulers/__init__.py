from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, _LRScheduler


def get_scheduler(name):
    if name is None:
        name = 'constant_lr'
    return {
        "constant_lr": ConstantLR,
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exp_lr": ExponentialLR,
    }[name]


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.optimizer.__class__.__name__)
