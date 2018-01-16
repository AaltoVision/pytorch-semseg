import torch
import torch.nn as nn
from  collections import OrderedDict
from ptsemseg.utils import initialize_weights

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        if bn_size is not None:
            self.add_module('norm.1', nn.BatchNorm2d(num_input_features))
            self.add_module('relu.1', nn.ReLU(inplace=True))
            self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))

            self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu.2', nn.ReLU(inplace=True))
            self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm.1', nn.BatchNorm2d(num_input_features))
            self.add_module('relu.1', nn.ReLU(inplace=True))
            self.add_module('conv.1', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('dropout', nn.Dropout2d(p=drop_rate))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))

class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('upconv', nn.ConvTranspose2d(num_input_features, num_output_features,
                                             kernel_size=4, stride=2, bias=False))

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, bn_size=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class tiramisu(nn.Module):
    def __init__(self, n_classes, num_init_features, growth_rate, encoder_cf, bottleneck_cf, decoder_cf):
        super(tiramisu, self).__init__()

        encoder_cf_parts = encoder_cf.split('-')
        decoder_cf_parts = decoder_cf.split('-')

        encoder_cf = [int(x) for x in encoder_cf_parts]
        decoder_cf = [int(x) for x in decoder_cf_parts]

        compression = 1
        bn_size = 1
        drop_rate = 0.2
        num_features = num_init_features
        self.num_encoder_blocks = len(encoder_cf)
        self.growth_rate = growth_rate
        self.n_classes = n_classes

        #First convolution (our backbone is same as a pure densenet)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        #Encoder denseblocks
        for i, num_layers in enumerate(encoder_cf):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
#            if i != len(encoder_cf) - 1:
            block = _TransitionDown(num_input_features=num_features,
                                    num_output_features=int(num_features * compression), drop_rate=drop_rate)
            self.features.add_module('transition-down%d' % (i + 1), block)
            num_features = int(num_features * compression)

        #Bottleneck in the middle
        block = _DenseBlock(num_layers=bottleneck_cf,
                            num_input_features=num_features,
                            growth_rate=growth_rate,
                            drop_rate=drop_rate)
        self.features.add_module('bottleneck', block)

        num_features = bottleneck_cf * growth_rate

        #The first transposed convolution
        block = _TransitionUp(num_input_features=num_features,
                                num_output_features=num_features)
        self.features.add_module('transition-up1', block)

        total = 0
        sums = []
        for v in encoder_cf:
            total+=v
            sums.append(total)
        num_features_shortcuts = [num_init_features + x * growth_rate for x in sums]
        num_features_shortcuts = num_features_shortcuts[::-1]
        temp = [bottleneck_cf] + decoder_cf
        temp = temp[:-1]
        num_features_from_below = [x * growth_rate for x in temp]

        #number of layers to be fed to each layer in decoder
        num_features_dec = [sum(x) for x in zip(num_features_shortcuts, num_features_from_below)]
        num_features_from_below = num_features_from_below[1:]

        #Decoder denseblocks
        for i, num_layers in enumerate(decoder_cf):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features_dec[i],
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock-up%d' % (i + 1), block)
            if i != len(decoder_cf) - 1:
                block = _TransitionUp(num_input_features=num_features_from_below[i],
                                        num_output_features=num_features_from_below[i])
                self.features.add_module('transition-up%d' % (i + 2), block)
        filters = num_features_from_below[-1] + num_features_shortcuts[-1] + decoder_cf[-1] * growth_rate
        block = nn.Conv2d(filters, n_classes, 1)
        self.features.add_module('predictionlayer', block)
        initialize_weights(self.features)

    def forward(self, x):
        x = self.features[0](x) # Very first convolution

        keep_shortcuts = []

        x = self.features[1](x) # Denseblock-down1
        keep_shortcuts.append(x)
        x = self.features[2](x) # Transition-down1

        x = self.features[3](x) # Denseblock-down2
        keep_shortcuts.append(x)
        x = self.features[4](x) # Transition-down2

        x = self.features[5](x) # Denseblock-down3
        keep_shortcuts.append(x)
        x = self.features[6](x) # Transition-down3

        x = self.features[7](x) # Denseblock-down4
        keep_shortcuts.append(x)
        x = self.features[8](x) # Transition-down4

        x = self.features[9](x) # Denseblock-down5
        keep_shortcuts.append(x)
        x = self.features[10](x) # Transition-down5

        keep_shortcuts = keep_shortcuts[::-1]

        keep = []
        for name, layer in self.features[11].named_children():
            x = layer(x)
            keep.append(x.narrow(1,0, self.growth_rate))

        x = self.features[12](torch.cat(keep,1)) # Transition-up1
        x = torch.cat((x[:,:,1:-1,1:-1], keep_shortcuts[0]),1)

        del keep[:]
        for name, layer in self.features[13].named_children():
            x = layer(x)
            keep.append(x.narrow(1,0, self.growth_rate))

        x = self.features[14](torch.cat(keep,1)) # Transition-up2)
        x = torch.cat((x[:,:,1:-1,1:-1], keep_shortcuts[1]),1)

        del keep[:]
        for name, layer in self.features[15].named_children():
            x = layer(x)
            keep.append(x.narrow(1,0, self.growth_rate))

        x = self.features[16](torch.cat(keep,1)) # Transition-up3
        x = torch.cat((x[:,:,1:-1,1:-1], keep_shortcuts[2]),1)

        del keep[:]
        for name, layer in self.features[17].named_children():
            x = layer(x)
            keep.append(x.narrow(1,0, self.growth_rate))

        x = self.features[18](torch.cat(keep,1)) # Transition-up4
        x = torch.cat((x[:,:,1:-1,1:-1], keep_shortcuts[3]),1)

        del keep[:]
        for name, layer in self.features[19].named_children():
            x = layer(x)
            keep.append(x.narrow(1,0, self.growth_rate))

        x = self.features[20](torch.cat(keep,1)) # Transition-up5
        x = torch.cat((x[:,:,1:-1,1:-1], keep_shortcuts[4]),1)
        x = self.features[21](x)
        x = self.features[22](x) # Final layer 1x1 conv

        #x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes)

#        out = nn.functional.log_softmax(x)
        out = x
        return out
