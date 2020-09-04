from collections import OrderedDict

from torch import nn

from .resnet import get_resnet_model
from .tools import conv1x1, conv3x3, DecoderModule, get_norm_layer, UpsampleCatConv
from utils.logger import print_info, print_warning

INPUT_CHANNELS = 3
FINAL_LAYER_CHANNELS = 32
LAYER1_REDUCED_CHANNELS = 128
LAYER2_REDUCED_CHANNELS = 256
LAYER3_REDUCED_CHANNELS = 512
LAYER4_REDUCED_CHANNELS = 1024


class ResUNet(nn.Module):
    """U-Net with residual encoder backbone."""

    @property
    def name(self):
        return self.enc_name.replace('res', 'res_u')

    def __init__(self, n_classes, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.norm_layer_kwargs = kwargs.pop('norm_layer', dict())
        self.norm_layer = get_norm_layer(**self.norm_layer_kwargs)
        self.no_maxpool = kwargs.get('no_maxpool', False)
        self.conv_as_maxpool = kwargs.get('conv_as_maxpool', True)
        self.use_upcatconv = kwargs.get('use_upcatconv', False)
        self.use_deconv = kwargs.get('use_deconv', False)
        assert not (self.use_deconv and self.use_upcatconv)
        self.same_up_channels = kwargs.get('same_up_channels', False)
        self.use_conv1x1 = kwargs.get('use_conv1x1', False)
        assert not (self.conv_as_maxpool and self.no_maxpool)
        self.enc_name = kwargs.get('encoder_name', 'resnet18')
        self.reduced_layers = kwargs.get('reduced_layers', False) and self.enc_name not in ['resnet18, resnet34']

        pretrained = kwargs.get('pretrained_encoder', True)
        replace_with_dilation = kwargs.get('replace_with_dilation')
        strides = kwargs.get('strides', 2)
        resnet = get_resnet_model(self.enc_name)(pretrained, progress=False, norm_layer=self.norm_layer_kwargs,
                                                 strides=strides, replace_with_dilation=replace_with_dilation)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # XXX: maxpool creates high amplitude high freq activations, removing it leads to better results
        if self.conv_as_maxpool:
            layer0_out_channels = self.get_nb_out_channels(self.layer0)
            self.layer1 = nn.Sequential(*[conv3x3(layer0_out_channels, layer0_out_channels, stride=2),
                                          self.norm_layer(layer0_out_channels),
                                          nn.ReLU()] + list(resnet.layer1.children()))
        elif self.no_maxpool:
            self.layer1 = nn.Sequential(*list(resnet.layer1.children()))
        else:
            self.layer1 = nn.Sequential(*[resnet.maxpool] + list(resnet.layer1.children()))
        self.layer2, self.layer3, self.layer4 = resnet.layer2, resnet.layer3, resnet.layer4

        layer0_out_channels = self.get_nb_out_channels(self.layer0)
        layer1_out_channels = self.get_nb_out_channels(self.layer1)
        layer2_out_channels = self.get_nb_out_channels(self.layer2)
        layer3_out_channels = self.get_nb_out_channels(self.layer3)
        layer4_out_channels = self.get_nb_out_channels(self.layer4)
        if self.reduced_layers:
            self.layer1_red = self._reducing_layer(layer1_out_channels, LAYER1_REDUCED_CHANNELS)
            self.layer2_red = self._reducing_layer(layer2_out_channels, LAYER2_REDUCED_CHANNELS)
            self.layer3_red = self._reducing_layer(layer3_out_channels, LAYER3_REDUCED_CHANNELS)
            self.layer4_red = self._reducing_layer(layer4_out_channels, LAYER4_REDUCED_CHANNELS)
            layer1_out_channels, layer2_out_channels = LAYER1_REDUCED_CHANNELS, LAYER2_REDUCED_CHANNELS
            layer3_out_channels, layer4_out_channels = LAYER3_REDUCED_CHANNELS, LAYER4_REDUCED_CHANNELS

        self.layer4_up = self._upsampling_layer(layer4_out_channels, layer3_out_channels, layer3_out_channels)
        self.layer3_up = self._upsampling_layer(layer3_out_channels, layer2_out_channels, layer2_out_channels)
        self.layer2_up = self._upsampling_layer(layer2_out_channels, layer1_out_channels, layer1_out_channels)
        self.layer1_up = self._upsampling_layer(layer1_out_channels, layer0_out_channels, layer0_out_channels)
        self.layer0_up = self._upsampling_layer(layer0_out_channels, FINAL_LAYER_CHANNELS, INPUT_CHANNELS)
        self.final_layer = self._final_layer(FINAL_LAYER_CHANNELS)

        if not pretrained:
            self._init_conv_weights()

        print_info("Model {} initialisated with norm_layer={}({}) and kwargs {}"
                   .format(self.name, self.norm_layer.func.__name__, self.norm_layer.keywords, kwargs))

    def _reducing_layer(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict([
            ('conv', conv1x1(in_channels, out_channels)),
            ('bn', self.norm_layer(out_channels)),
            ('relu', nn.ReLU()),
        ]))

    def get_nb_out_channels(self, layer):
        return list(filter(lambda e: isinstance(e, nn.Conv2d), layer.modules()))[-1].out_channels

    def _upsampling_layer(self, in_channels, out_channels, cat_channels):
        if self.use_upcatconv:
            return UpsampleCatConv(in_channels + cat_channels, out_channels, norm_layer=self.norm_layer,
                                   use_conv1x1=self.use_conv1x1)
        else:
            up_channels = in_channels if self.same_up_channels else None
            return DecoderModule(in_channels, out_channels, cat_channels, up_channels=up_channels,
                                 norm_layer=self.norm_layer, n_conv=1, use_deconv=self.use_deconv,
                                 use_conv1x1=self.use_conv1x1)

    def _final_layer(self, in_channels):
        return nn.Sequential(OrderedDict([('conv', conv1x1(in_channels, self.n_classes))]))

    def _init_conv_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state and state[name].shape == param.shape:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)
            else:
                unloaded_params.append(name)

        if len(unloaded_params) > 0:
            print_warning('load_state_dict: {} not found'.format(unloaded_params))

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.reduced_layers:
            x4 = self.layer4_red(x4)
            x3 = self.layer3_red(x3)
            x2 = self.layer2_red(x2)
            x1 = self.layer1_red(x1)

        x3 = self.layer4_up(x4, other=x3)
        x2 = self.layer3_up(x3, other=x2)
        x1 = self.layer2_up(x2, other=x1)
        x0 = self.layer1_up(x1, other=x0)
        x = self.layer0_up(x0, other=x)
        x = self.final_layer(x)

        return x
