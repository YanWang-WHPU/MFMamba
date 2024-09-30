from mmcv.cnn import ConvModule, build_norm_layer
from typing import Optional, Union, Sequence
# from mmrotate.models.utils import  make_divisible
from mmengine.model import BaseModule

#### PKIblock
class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 1, 1, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels
        hidden_channels = out_channels
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg) # 1x1

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  1,  dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None) 
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   2, dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   3, dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   4, dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   5, dilations[4],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg) # 1x1 填充为0 步长为1

        # if with_caa:
        #     self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        # else:
        #     self.caa_factor = None

        # self.add_identity = add_identity and in_channels == out_channels

        # self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
        #                             norm_cfg=norm_cfg, act_cfg=act_cfg)
   ## PKI模块
    def forward(self, x):
        x = self.pre_conv(x) #1x1

        # y = x.clone() # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x) #3X3
        # x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x) # 5个处理后的卷积相加
        x = x  + self.dw_conv1(x) + self.dw_conv2(x) #  3X3 1X1 5X5
        x = self.pw_conv(x) # 1x1卷积
        # if self.caa_factor is not None:
        #     y = self.caa_factor(y)
        # if self.add_identity:
        #     y = x * y
        #     x = x + y
        # else:
        #     x = x * y

        # x = self.post_conv(x)
        return x
    
