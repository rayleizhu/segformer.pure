import numpy as np
import torch.nn as nn
import torch
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from torch import Tensor
from typing import List
# from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from mmseg.models.utils import *
from collections import OrderedDict
# import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x:Tensor)->Tensor:
        # equivalent to conv1x1?
        x = x.flatten(2).transpose(1, 2) # (B, C, H, W) -> (B, H*W, C)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes, in_index, in_channels, feature_strides, embedding_dim, dropout_ratio):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        # it seems that, feature_strides is not used to define modules, just a record
        assert min(feature_strides) == feature_strides[0]
        
        self.in_index = in_index
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/conv_module.py
        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='SyncBN', requires_grad=True)
        # )
        # TODO: check other default args
        self.linear_fuse = nn.Sequential(
            OrderedDict(conv=nn.Conv2d(in_channels=embedding_dim*4,
                                       out_channels=embedding_dim,
                                       kernel_size=1,
                                       bias=False),
                        bn=nn.BatchNorm2d(num_features=embedding_dim),
                        act=nn.ReLU(inplace=True)
            )     
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs:List[Tensor]):
        x = [inputs[i] for i in self.in_index]  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # https://github.com/NVlabs/SegFormer/blob/master/mmseg/ops/wrappers.py
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x