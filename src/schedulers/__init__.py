from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, _LRScheduler


def get_scheduler(name):
    if name is None:
        name = 'constant_lr'
    return {
        "constant_lr": ConstantLR,
        "poly_lr": PolynomialLR,
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


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]

    def __str__(self):
        params = [
            'optimizer: {}'.format(self.optimizer.__class__.__name__),
            'decay_iter: {}'.format(self.decay_iter),
            'max_iter: {}'.format(self.max_iter),
            'gamma: {}'.format(self.gamma),
        ]
        return '{}({})'.format(self.__class__.__name__, ','.join(params))
