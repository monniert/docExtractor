from collections import OrderedDict
from functools import partial

from torch import nn, cat
from torch.nn import functional as F


def get_norm_layer(**kwargs):
    name = kwargs.get('name', 'batch_norm')
    momentum = kwargs.get('momentum', 0.1)
    affine = kwargs.get('affine', True)
    track_stats = kwargs.get('track_running_stats', True)
    num_groups = kwargs.get('num_groups', 32)

    norm_layer = {
        'batch_norm': partial(nn.BatchNorm2d, momentum=momentum, affine=affine, track_running_stats=track_stats),
        'group_norm': partial(nn.GroupNorm, num_groups=num_groups, affine=affine),
        'instance_norm': partial(nn.InstanceNorm2d, momentum=momentum, affine=affine, track_running_stats=track_stats),
    }[name]
    if norm_layer.func == nn.GroupNorm:
        return lambda num_channels: norm_layer(num_channels=num_channels)
    else:
        return norm_layer


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_model_state_dict(state_dict):
    """Convert a state dict saved from a DataParallel module to normal module state_dict."""
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v  # remove 'module.' prefix
    return new_state_dict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class UpsampleCatConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, mode='bilinear', use_conv1x1=False):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        conv_layer = conv1x1 if use_conv1x1 else conv3x3
        self.mode = mode
        self.conv = conv_layer(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, other):
        x = nn.functional.interpolate(x, size=(other.size(2), other.size(3)), mode=self.mode, align_corners=False)
        x = cat((x, other), dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, mode='bilinear', use_conv1x1=False):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        conv_layer = conv1x1 if use_conv1x1 else conv3x3
        self.mode = mode
        self.conv = conv_layer(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, output_size):
        x = nn.functional.interpolate(x, size=output_size[2:], mode=self.mode, align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DeconvModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, output_size):
        x = self.deconv(x, output_size=output_size)
        x = self.norm(x)
        x = self.act(x)
        return x


class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, cat_channels=None, up_channels=None,
                 norm_layer=None, n_conv=2, use_deconv=False, use_conv1x1=False):
        super().__init__()
        cat_channels = cat_channels or in_channels // 2
        up_channels = up_channels or in_channels // 2
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        self.use_deconv = use_deconv
        if use_deconv:
            self.decode = DeconvModule(in_channels, up_channels, norm_layer)
        else:
            self.decode = UpsampleConv(in_channels, up_channels, norm_layer, 'bilinear', use_conv1x1)
        self.conv_block = nn.Sequential(OrderedDict(sum([[
            ('conv{}'.format(k + 1), conv3x3(up_channels + cat_channels if k == 0 else out_channels, out_channels)),
            ('bn{}'.format(k + 1), norm_layer(out_channels)),
            ('relu{}'.format(k + 1), nn.ReLU())]
            for k in range(n_conv)], [])))

    def forward(self, x, other):
        try:
            x = self.decode(x, output_size=other.size())
        except ValueError:
            # XXX a size adjustement is needed for odd sizes
            B, C, H, W = other.size()
            h, w = H // 2 * 2, W // 2 * 2
            x = self.decode(x, output_size=(B, C, h, w))
            x = F.pad(x, (W - w, 0, H - h, 0))
        x = cat((x, other), dim=1)
        x = self.conv_block(x)
        return x
