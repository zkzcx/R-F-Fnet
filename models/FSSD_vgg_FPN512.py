import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import vgg, vgg_base

class LDS(nn.Module):
    def __init__(self,):
        super(LDS, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        return x_pool3


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LSN_init(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_init, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = nn.Sequential(
                ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=1, stride=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
                )
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2


class LSN_later(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_later, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2







class Relu_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Relu_Conv, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1)
        )

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out




class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear',align_corners=True) if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class FSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1712.00960.pdf or more details.
    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base, extras, ft_module, pyramid_ext, head, num_classes, size):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.ft_module = nn.ModuleList(ft_module)
        self.pyramid_ext = nn.ModuleList(pyramid_ext)
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)

        self.lds = LDS()

        # convs for merging the lsn and ssd features
        self.Norm1 = Relu_Conv(512, 512, stride=1)
        self.Norm2 = Relu_Conv(512, 512, stride=1)
        self.Norm3 = Relu_Conv(256, 256, stride=1)
        self.Norm4 = Relu_Conv(256, 256, stride=1)
        self.Norm5 = Relu_Conv(256, 256, stride=1)

        # convs for generate the lsn features
        self.icn1 = LSN_init(3, 512, stride=1)
        self.icn2 = LSN_later(128, 512, stride=2)
        self.icn3 = LSN_later(128, 256, stride=2)
        # self.icn3 = LSN_later(256, 512, stride=2)
        self.icn4 = LSN_later(64, 256, stride=2)
        self.icn5 = LSN_later(64, 256, stride=2)

        

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()

        self.conv_cat0 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.upsample0 = nn.Upsample(size=(2, 2), mode='bilinear',align_corners=True)

        self.conv_cat1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.upsample1 = nn.Upsample(size=(4, 4), mode='bilinear',align_corners=True)

        self.conv_cat2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.upsample2 = nn.Upsample(size=(8, 8), mode='bilinear',align_corners=True)

        self.conv_cat3 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.upsample3 = nn.Upsample(size=(16, 16), mode='bilinear',align_corners=True)

        self.conv_cat4 = nn.Conv2d(768, 512, kernel_size=1, padding=0, stride=1)
        self.upsample4 = nn.Upsample(size=(32, 32), mode='bilinear',align_corners=True)

        self.conv_cat5 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)
        self.upsample5 = nn.Upsample(size=(64, 64), mode='bilinear',align_corners=True)

        # self.convcat0 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)
        # self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,ceil_mode=True)
        # #
        # self.convcat1 = nn.Conv2d(768, 256, kernel_size=1, padding=0, stride=1)
        # self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,ceil_mode=False)
        # #
        # #
        # self.convcat2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        # #
        # #
        # self.convcat3 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        # #
        # #
        # self.convcat4 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        #
        # self.convcat5 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        # self.con0 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        # #self.con1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        # self.con2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        # #self.con3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        # #self.con4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        source_features = list()
        transformed_features = list()
        loc = list()
        conf = list()

        # apply lds to the initial image
        x_pool = self.lds(x)

        # # apply vgg up to conv4_3
        # for k in range(22):
        #     x = self.base[k](x)
        # conv4_3_bn = self.ibn1(x)
        x_pool1_skip, x_pool1_icn = self.icn1(x_pool)
        # s = self.Norm1(conv4_3_bn * x_pool1_icn)

        # # apply vgg up to fc7
        # for k in range(22, 34):
        #     x = self.base[k](x)
        # conv7_bn = self.ibn2(x)
        x_pool2_skip, x_pool2_icn = self.icn2(x_pool1_skip)
        # p = self.Norm2(self.dsc1(s) + conv7_bn * x_pool2_icn)

        x_pool3_skip, x_pool3_icn = self.icn3(x_pool2_skip)
        x_pool4_skip, x_pool4_icn = self.icn4(x_pool3_skip)
        x_pool5_skip, x_pool5_icn = self.icn5(x_pool4_skip)
        # x_pool4_skip, x_pool4_icn = self.icn4(x_pool2_skip)

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        source_features.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        source_features.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
        source_features.append(x)
        assert len(self.ft_module) == len(source_features)
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(source_features[k]))
        concat_fea = torch.cat(transformed_features, 1)
        x = self.fea_bn(concat_fea)
        pyramid_fea = list()
        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x)
        
        pyramid_fea[0] = pyramid_fea[0] + x_pool1_icn
        pyramid_fea[1] = pyramid_fea[1] + x_pool2_icn
        pyramid_fea[2] = pyramid_fea[2] + x_pool3_icn
        pyramid_fea[3] = pyramid_fea[3] + x_pool4_icn
        pyramid_fea[4] = pyramid_fea[4] + x_pool5_icn

        pyramid_fea[0] = self.Norm1(pyramid_fea[0])
        pyramid_fea[1] = self.Norm2(pyramid_fea[1])
        pyramid_fea[2] = self.Norm3(pyramid_fea[2])
        pyramid_fea[3] = self.Norm4(pyramid_fea[3])
        pyramid_fea[4] = self.Norm4(pyramid_fea[4])


        fpn0 = pyramid_fea[0]
        fpn1 = pyramid_fea[1]
        fpn2 = pyramid_fea[2]
        fpn3 = pyramid_fea[3]
        fpn4 = pyramid_fea[4]
        fpn5 = pyramid_fea[5]
        fpn6 = pyramid_fea[6]


        # ----------this block is to downsample the 1*1 layer to 3*3, and concat with the original 3*3 layer, like Dense connection
        fpn_0 = list()
        detect_6 = pyramid_fea[6]
        detect_5 = pyramid_fea[5]
        detect_6_5 = self.upsample0(detect_6)
        #detect_5_4 = Inception_A(256)(detect_5_4)
        fpn_0.append(detect_5)
        fpn_0.append(detect_6_5)
        detect_5 = torch.cat(fpn_0, 1)
        detect_5 = self.conv_cat0(detect_5)
        pyramid_fea[5] = detect_5
        pyramid_fea[6] = detect_6



        # ----------this block is to downsample the 3*3 layer to 5*5, and concat with the original 5*5 layer, like Dense connection
        fpn_1 = list()
        detect_4 = pyramid_fea[4]
        detect_5_4 = self.upsample1(detect_5)
        #detect_4_3 = Inception_A(256)(detect_4_3)
        fpn_1.append(detect_4)
        fpn_1.append(detect_5_4)
        detect_4 = torch.cat(fpn_1, 1)
        detect_4 = self.conv_cat1(detect_4)
        pyramid_fea[4] = detect_4

        # ----------this block is to downsample the 5*5 layer to 10*10, and concat with the original 10*10 layer, like Dense connection
        fpn_2 = list()
        detect_3 = pyramid_fea[3]
        detect_4_3 = self.upsample2(detect_4)
        #detect_3_2 = Inception_A(256)(detect_3_2)
        fpn_2.append(detect_3)
        fpn_2.append(detect_4_3)
        detect_3 = torch.cat(fpn_2, 1)
        detect_3 = self.conv_cat2(detect_3)
        pyramid_fea[3] = detect_3

        # ----------this block is to downsample the 10*10 layer to 19*19, and concat with the original 19*19 layer, like Dense connection
        fpn_3 = list()
        detect_2 = pyramid_fea[2]
        detect_3_2 = self.upsample3(detect_3)
        #detect_2_1 = Inception_A(256)(detect_2_1)
        fpn_3.append(detect_2)
        fpn_3.append(detect_3_2)
        detect_2 = torch.cat(fpn_3, 1)
        detect_2 = self.conv_cat3(detect_2)
        pyramid_fea[2] = detect_2

        # ----------this block is to downsample the 10*10 layer to 19*19, and concat with the original 19*19 layer, like Dense connection
        fpn_4 = list()
        detect_1 = pyramid_fea[1]
        detect_2_1 = self.upsample4(detect_2)
        #detect_2_1 = Inception_A(256)(detect_2_1)
        fpn_4.append(detect_1)
        fpn_4.append(detect_2_1)
        detect_1 = torch.cat(fpn_4, 1)
        detect_1 = self.conv_cat4(detect_1)
        pyramid_fea[1] = detect_1

        # ----------this block is to downsample the 19*19 layer to 38*38, and concat with the original 38*38 layer, like Dense connection
        fpn_5 = list()
        detect_0 = pyramid_fea[0]
        detect_1_0 = self.upsample5(detect_1)
        #detect_1_0 = Inception_A(512)(detect_1_0)
        fpn_5.append(detect_0)
        fpn_5.append(detect_1_0)
        detect_0 = torch.cat(fpn_5, 1)
        detect_0 = self.conv_cat5(detect_0)
        pyramid_fea[0] = detect_0

       # # ----------this block is to downsample the 3*3 layer to 5*5, and concat with the original 5*5 layer, like Dense connection
       #  fpn_00 = list()
       #  detect0 = fpn0
       #  detect1 = fpn1
       #  detect_0_1 = self.downsample(detect0)
       #  #detect_0_1 = Inception_A(512)(detect_0_1)
       #  fpn_00.append(detect1)
       #  fpn_00.append(detect_0_1)
       #  detect1 = torch.cat(fpn_00, 1)
       #  detect1 = self.convcat0(detect1)
       #
       #  fpn_01 = list()
       #
       #  detect2 = fpn2
       #  detect_0_2 = self.downsample(detect1)
       #  #detect_0_2 = Inception_A(512)(detect_0_2)
       #  fpn_01.append(detect2)
       #  fpn_01.append(detect_0_2)
       #  detect2 = torch.cat(fpn_01, 1)
       #  detect2 = self.convcat1(detect2)
       #
       #  fpn_02 = list()
       #
       #  detect3 = fpn3
       #  detect_0_3 = self.downsample(detect2)
       #  #detect_0_3 = Inception_A(256)(detect_0_3)
       #  fpn_02.append(detect3)
       #  fpn_02.append(detect_0_3)
       #  detect3 = torch.cat(fpn_02, 1)
       #  detect3 = self.convcat2(detect3)
       #
       #  fpn_03 = list()
       #
       #  detect4 = fpn4
       #  detect_0_4 = self.downsample(detect3)
       #  #detect_0_4 = Inception_A(256)(detect_0_4)
       #  fpn_03.append(detect4)
       #  fpn_03.append(detect_0_4)
       #  detect4 = torch.cat(fpn_03, 1)
       #  detect4 = self.convcat3(detect4)
       #
       #  fpn_04 = list()
       #
       #  detect5 = fpn5
       #  detect_0_5 = self.downsample1(detect4)
       #  #detect_0_5 = Inception_A(256)(detect_0_5)
       #  fpn_04.append(detect5)
       #  fpn_04.append(detect_0_5)
       #  detect5 = torch.cat(fpn_04, 1)
       #  detect5 = self.convcat4(detect5)

        # # sources[0] = sources[0] +detect0
        # pyramid_fea[1] = pyramid_fea[1] + detect1
        # pyramid_fea[2] = pyramid_fea[2] + detect2
        # pyramid_fea[3] = pyramid_fea[3] + detect3
        # pyramid_fea[4] = pyramid_fea[4] + detect4
        # pyramid_fea[5] = detect5
        # pyramid_fea[0] = self.con0(pyramid_fea[0])
        # pyramid_fea[1] = self.con0(pyramid_fea[1])
        # pyramid_fea[2] = self.con2(pyramid_fea[2])
        # pyramid_fea[3] = self.con2(pyramid_fea[3])
        # pyramid_fea[4] = self.con2(pyramid_fea[4])
        

        # # ----------this block is to downsample the 5*5 layer to 10*10, and concat with the original 10*10 layer, like Dense connection
        # fpn_244 = list()
        # 
        # detect_3_24 = self.upsample2(x_pool4_icn)
        # #detect_3_2 = Inception_A(256)(detect_3_2)
        # fpn_244.append(x_pool3_icn)
        # fpn_244.append(detect_3_24)
        # detect_23 = torch.cat(fpn_244, 1)
        # x_pool3_icn = self.conv_cat2(detect_23)
        # 
        # 
        # # ----------this block is to downsample the 10*10 layer to 19*19, and concat with the original 19*19 layer, like Dense connection
        # fpn_333 = list()
        # detect_2_133 = self.upsample3(x_pool3_icn)
        # #detect_2_1 = Inception_A(256)(detect_2_1)
        # fpn_333.append(x_pool2_icn)
        # fpn_333.append(detect_2_133)
        # detect_133 = torch.cat(fpn_333, 1)
        # x_pool2_icn = self.conv_cat3(detect_133)
        # 
        # 
        # # ----------this block is to downsample the 19*19 layer to 38*38, and concat with the original 38*38 layer, like Dense connection
        # fpn_411 = list()
        # 
        # detect_1_022 = self.upsample4(x_pool2_icn)
        # #detect_1_0 = Inception_A(512)(detect_1_0)
        # fpn_411.append(x_pool1_icn)
        # fpn_411.append(detect_1_022)
        # detect_011 = torch.cat(fpn_411, 1)
        # x_pool1_icn = self.conv_cat4(detect_011)
        

      
        

        

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def feature_transform_module(vgg, extral, size):
    if size == 300:
        up_size = 38
    elif size == 512:
        up_size = 64

    layers = []
    # conv4_3
    layers += [BasicConv(vgg[24].out_channels, 256, kernel_size=1, padding=0)]
    # fc_7
    layers += [BasicConv(vgg[-2].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    layers += [BasicConv(extral[-1].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    return vgg, extral, layers


def pyramid_feature_extractor(size):
    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers


def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


extras = {
    '300': [256, 512, 128, 'S', 256],
    '512': [256, 512, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}
fea_channels = {
    '300': [512, 512, 256, 256, 256, 256],
    '512': [512, 512, 256, 256, 256, 256, 256]}


def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only FSSD300 and FSSD512 is supported currently!")
        return

    return FSSD(*feature_transform_module(vgg(vgg_base[str(size)], 3), add_extras(extras[str(size)], 1024), size=size),
                pyramid_ext=pyramid_feature_extractor(size),
                head=multibox(fea_channels[str(size)], mbox[str(size)], num_classes), num_classes=num_classes,
size=size)