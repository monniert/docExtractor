from functools import partial
import torch

from utils import coerce_to_path_and_check_exist

from .res_unet import ResUNet
from .tools import safe_model_state_dict


def get_model(name=None):
    if name is None:
        name = 'res_unet18'
    return {
        'res_unet18': partial(ResUNet, encoder_name='resnet18'),
        'res_unet34': partial(ResUNet, encoder_name='resnet34'),
        'res_unet50': partial(ResUNet, encoder_name='resnet50'),
        'res_unet101': partial(ResUNet, encoder_name='resnet101'),
        'res_unet152': partial(ResUNet, encoder_name='resnet152'),
    }[name]


def load_model_from_path(model_path, device=None, attributes_to_return=None, eval_mode=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(coerce_to_path_and_check_exist(model_path), map_location=device.type)
    checkpoint['model_kwargs']['pretrained_encoder'] = False
    model = get_model(checkpoint['model_name'])(checkpoint['n_classes'], **checkpoint['model_kwargs']).to(device)
    model.load_state_dict(safe_model_state_dict(checkpoint['model_state']))
    if eval_mode:
        model.eval()
    if attributes_to_return is not None:
        if isinstance(attributes_to_return, str):
            attributes_to_return = [attributes_to_return]
        return model, [checkpoint.get(key) for key in attributes_to_return]
    else:
        return model
