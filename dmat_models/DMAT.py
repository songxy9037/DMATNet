import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from networks import ResNet
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.ps_vit import ps_vit
from models.TR import TR
from cc_attention.functions import CrissCrossAttention
from MA.mixedatten import Block

class DMAT(nn.Module):

    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4,
                 backbone='resnet18',
                 x_w=64,
                 x_h=64,
                 num_ps=4,
                 num_ma=1
                 ):
        super(DMAT, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.resnet = ResNet(input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True)
        self.token_len = token_len
        self.psvit = ps_vit(num_iters=num_ps)
        self.x_w = x_w
        self.x_h = x_h
        self.cca = CrissCrossAttention(32)
        self.TR = TR(token_len=4, decdepth=4, in_channel=64)
        self.MAM_block = Block(dim_in=32,
                 dim_out=32,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 freeze_bn=False,
                 )
    def ps_layer(self, x):
        b = x.shape[0]
        c = 32
        p = self.psvit(x)
        p = rearrange(p, 'b s c -> b c s')
        p = p.view([b, c, self.x_w, self.x_h])
        return p

    def Backbone(self, x1, x2):

        p1 = self.ps_layer(x1)
        p2 = self.ps_layer(x2)
        # forward backbone resnet
        x1 = self.resnet(x1)    # 32,64,64
        x2 = self.resnet(x2)
        return p1, p2, x1, x2

    def MAM(self, p, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c w h -> b (w h) c')
        p = rearrange(p, 'b c w h -> b (w h) c')
        cat = torch.cat([p, x], dim=1)
        cat = self.MAM_block(cat, w, h, h, w)
        p, x = torch.split(cat, [h * w, h * w], dim=1)
        p = p.view([b, c, w, h])

        return p

    def forward(self, x1, x2):

        # Backbone dual feature
        p1, p2, x1, x2 = self.Backbone(x1, x2)

        # mixed anntneion module
        p1 = self.MAM(p1, x1)
        p2 = self.MAM(p2, x2)

        # transformer module
        X_1, X_2 = self.TR(x1,x2,p1,p2)

        # feature differencing
        x = torch.abs(X_1 - X_2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)

        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)

        return x