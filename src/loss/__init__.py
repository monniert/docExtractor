from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def get_loss(name):
    if name is None:
        name = 'bce'
    return {
        'bce': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
    }[name]
