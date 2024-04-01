import torch
import torch.nn as nn
import torch.nn.functional as F
import math

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'SepCDC_3x3_0.0': lambda C, stride, affine: SepCDC_theta(C, C, 3, stride, 1, theta=0.0, affine=affine),
    'SepCDC_3x3_0.5': lambda C, stride, affine: SepCDC_theta(C, C, 3, stride, 1, theta=0.5, affine=affine),
    'SepCDC_3x3_0.7': lambda C, stride, affine: SepCDC_theta(C, C, 3, stride, 1, theta=0.7, affine=affine),
    'SepCDC_3x3_0.8': lambda C, stride, affine: SepCDC_theta(C, C, 3, stride, 1, theta=0.8, affine=affine),
    'DilCDC_3x3_0.0': lambda C, stride, affine: DilCDC_theta(C, C, 3, stride, 2, 2, theta=0.0, affine=affine),
    'DilCDC_3x3_0.5': lambda C, stride, affine: DilCDC_theta(C, C, 3, stride, 2, 2, theta=0.5, affine=affine),
    'DilCDC_3x3_0.7': lambda C, stride, affine: DilCDC_theta(C, C, 3, stride, 2, 2, theta=0.0, affine=affine),
    'DilCDC_3x3_0.8': lambda C, stride, affine: DilCDC_theta(C, C, 3, stride, 2, 2, theta=0.8, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),


    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affineCDC: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'CDC_conv_3x3': lambda C, stride, affine: CDC_Conv(C, C, 3, stride, 1, theta=0.7, affine=affine),
    'CDC_k3_e1_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                          padding=1, expand_ratio=1, theta=0.7),
    'CDC_k3_e1_g2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=2, kernel_size=3,
                                                          padding=1, expand_ratio=1, theta=0.7),
    'CDC_k3_e3_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                          padding=1, expand_ratio=3, theta=0.7),
    'CDC_k3_e6_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                          padding=1, expand_ratio=6, theta=0.7),
    'CDC_k5_e1_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=5,
                                                          padding=2, expand_ratio=1, theta=0.7),
    'CDC_k5_e1_g2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=2, kernel_size=5,
                                                          padding=2, expand_ratio=1, theta=0.7),
    'CDC_k5_e3_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=5,
                                                          padding=2, expand_ratio=3, theta=0.7),
    'CDC_k5_e6_g1': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=5,
                                                          padding=2, expand_ratio=6, theta=0.7),
    'CDC_k3_e1_g1_d2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                          padding=2, expand_ratio=1, dilation=2, theta=0.7),
    'CDC_k3_e1_g2_d2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=2, kernel_size=3,
                                                             padding=2, expand_ratio=1, dilation=2, theta=0.7),
    'CDC_k3_e3_g1_d2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                             padding=2, expand_ratio=3, dilation=2, theta=0.7),
    'CDC_k3_e6_g1_d2': lambda C, stride, affine: CDC_Conv_r2(C, C, stride, group=1, kernel_size=3,
                                                             padding=2, expand_ratio=6, dilation=2, theta=0.7),
}




class InvertedResidualOperation(nn.Module):
    def __init__(self, C_in, C_out, stride, group, kernel_size, padding, expand_ratio, dilation=1):
        super(InvertedResidualOperation, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.group = group
        self.kernel_size = kernel_size
        self.padding = padding
        self.expand_ratio = expand_ratio
        self.dilation = dilation
        self.stride = stride
        hidden_dim = round(self.C_in* expand_ratio)
        self.op = nn.Sequential(
            nn.Conv2d(self.C_in, hidden_dim, kernel_size=1, stride=1, padding=0, groups=group, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=padding, dilation=dilation,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, C_out, kernel_size=1, stride=1, padding=0, groups=group, bias=False),
            nn.BatchNorm2d(C_out),
        )
    def forward(self, x):
        return self.op(x)




class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)



class SepCDC_theta(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, theta=0.0, affine=True):
        super(SepCDC_theta, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2d_cd(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False, theta=theta),
            Conv2d_cd(in_channels=C_in, out_channels=C_in, kernel_size=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            Conv2d_cd(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in,
                      bias=False, theta=theta),
            Conv2d_cd(in_channels=C_in, out_channels=C_out, kernel_size=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilCDC_theta(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, theta=0.7, affine=True):
        super(DilCDC_theta, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2d_cd(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=C_in, bias=False, theta=theta),
            Conv2d_cd(in_channels=C_in, out_channels=C_out, kernel_size=1, padding=0, bias=False, theta=theta),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)




class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out





class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class CDC_Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, theta=0.7, affine=True):
        super(CDC_Conv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.theta = theta

        self.op = nn.Sequential(
            Conv2d_cd(in_channels=self.C_in, out_channels=self.C_out, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, bias=False, theta=self.theta),
            nn.BatchNorm2d(self.C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class CDC_Conv_r2(nn.Module):
    def __init__(self, C_in, C_out, stride, group, kernel_size, padding, expand_ratio, dilation=1, theta=0.7):
        super(CDC_Conv_r2, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.group = group
        self.kernel_size = kernel_size
        self.padding = padding
        self.expand_ratio = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.theta = theta
        hidden_dim = round(self.C_in * expand_ratio)
        self.op = nn.Sequential(

            Conv2d_cd(self.C_in, hidden_dim, kernel_size=1, stride=1, padding=0, groups=group, bias=False, theta=self.theta),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            Conv2d_cd(hidden_dim, hidden_dim, kernel_size, stride, padding=self.padding, dilation=dilation,
                      groups=hidden_dim, bias=False, theta=self.theta),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            Conv2d_cd(hidden_dim, C_out, kernel_size=1, stride=1, padding=0, groups=group,bias=False, theta=self.theta),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)