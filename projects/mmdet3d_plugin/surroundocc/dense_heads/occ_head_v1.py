# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss, \
    multiscale_supervision2d, make2dgt, sem_scal_loss2d, geo_scal_loss2d
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # Query
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # Key
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # Value
        self.softmax = nn.Softmax(dim=-1)
        self.scale = embed_dim ** 0.5

    def forward(self, query, key, value):
        # Linear projections
        Q = self.query_proj(query)  # [Batch, Seq_Q, Embed_Dim]
        K = self.key_proj(key)      # [Batch, Seq_K, Embed_Dim]
        V = self.value_proj(value)  # [Batch, Seq_K, Embed_Dim]

        # Attention weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = self.softmax(attention_scores)

        # Weighted sum
        output = torch.matmul(attention_weights, V)
        return output
@HEADS.register_module()
class OccHead_v1(nn.Module):
    def __init__(self,
                 *args,
                 transformer_template=None,
                 num_classes=17,
                 volume_h=200,
                 volume_w=200,
                 volume_z=16,
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 out_dim=None,
                 **kwargs):
        super(OccHead_v1, self).__init__()
        self.conv_input = conv_input
        self.conv_output = conv_output

        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.img_channels = img_channels

        self.use_semantic = use_semantic
        self.embed_dims = embed_dims

        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        self.transformer_template = transformer_template
        self.out_dim = out_dim
        self._init_layers()

    def _init_layers(self):
        self.transformer = nn.ModuleList()
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            transformer.embed_dims = transformer.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]

            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]

            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)

        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg = dict(type='deconv3d', bias=False)
        conv_cfg = dict(type='Conv3d', bias=False)

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))

            self.deblocks.append(deblock)

        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)

        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))

        self.transfer_conv = nn.ModuleList()
        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg = dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                conv_cfg,
                in_channels=self.img_channels[i],
                out_channels=self.embed_dims[i],
                kernel_size=1,
                stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                                           nn.ReLU(inplace=True))

            self.transfer_conv.append(transfer_block)

        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg = dict(type='deconv', bias=False)
        conv_cfg = dict(type='Conv2d', bias=False)

        self.deblocks2d = nn.ModuleList()
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            self.deblocks2d.append(deblock)

        self.avg_pool1 = nn.AvgPool3d(kernel_size=(1, self.volume_h[0], 1), stride=(1, self.volume_h[0], 1))
        self.avg_pool2 = nn.AvgPool3d(kernel_size=(1, self.volume_h[1], 1), stride=(1, self.volume_h[1], 1))
        self.avg_pool3 = nn.AvgPool3d(kernel_size=(1, self.volume_h[2], 1), stride=(1, self.volume_h[2], 1))
        self.Cross_att = nn.ModuleList(
            [
                CrossAttention(embed_dim) for embed_dim in self.out_dim
            ]
        )
        self.occ2d = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ_2d = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ2d.append(occ_2d)
            else:
                occ_2d = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ2d.append(occ_2d)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()

        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        # 1 6
        # print("bs:")
        # print(bs)
        dtype = mlvl_feats[0].dtype

        volume_embed = []
        for i in range(self.fpn_level):
            volume_queries = self.volume_embedding[i].weight.to(dtype)

            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            _, _, C, H, W = mlvl_feats[i].shape
            view_features = self.transfer_conv[i](mlvl_feats[i].reshape(bs * num_cam, C, H, W)).reshape(bs, num_cam, -1,
                                                                                                        H, W)

            volume_embed_i = self.transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas
            )
            volume_embed.append(volume_embed_i)

        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            volume_embed_reshape_i = volume_embed[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2,
                                                                                                           1)

            volume_embed_reshape.append(volume_embed_reshape_i)

        outputs = []
        for2dfeat = volume_embed_reshape.copy()
        result = volume_embed_reshape.pop()
        #3d deblocks
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)
            # print("resulut:")
            # print(result.shape)
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp


        outputs_2d = []
        for2dfeat[0] = self.avg_pool1(for2dfeat[0])
        batch_size, channels, dim1, dim2, dim3 = for2dfeat[0].size()
        for2dfeat[0] = for2dfeat[0].view(batch_size, channels, dim1, dim3)

        for2dfeat[1] = self.avg_pool2(for2dfeat[1])
        batch_size, channels, dim1, dim2, dim3 = for2dfeat[1].size()
        for2dfeat[1] = for2dfeat[1].view(batch_size, channels, dim1, dim3)

        for2dfeat[2] = self.avg_pool3(for2dfeat[2])
        batch_size, channels, dim1, dim2, dim3 = for2dfeat[2].size()
        for2dfeat[2] = for2dfeat[2].view(batch_size, channels, dim1, dim3)
        result = for2dfeat.pop()
        for i in range(len(self.deblocks2d)):
            result = self.deblocks2d[i](result)
            # print("resulut:")
            # print(result.shape)
            if i in self.out_indices:
                outputs_2d.append(result)
            elif i < len(self.deblocks2d) - 2:  # we do not add skip connection at level 0
                volume_embed_temp = for2dfeat.pop()
                result = result + volume_embed_temp
        for i,feature in  enumerate(outputs_2d):
            batch_size, channels, dim1,  dim3 = feature.size()
            outputs_2d[i] = feature.view(batch_size, channels, dim1, 1, dim3)

        exp_2d_feats = []
        for i,feat in enumerate(outputs):
            batch_size, channels, dim1, dim2, dim3 = feat.size()
            exp_2d_feats.append(outputs_2d[i].permute(0,2,3,4,1).expand(-1, -1, dim2, -1, -1))
            # .expand(-1, -1, dim2, -1, -1)
        cross_result = []
        for i ,cross in enumerate(self.Cross_att):
            cross_result.append(cross(exp_2d_feats[i],outputs[i].permute(0,2,3,4,1),outputs[i].permute(0,2,3,4,1)).permute(0,4,1,2,3))

        occ_preds = []
        for i in range(len(outputs)):
            occ_pred = self.occ[i](cross_result[i])
            occ_preds.append(occ_pred)

        occ2d_preds = []
        for i in range(len(outputs)):
            batch_size, channels, dim1, dim2, dim3 = outputs_2d[i].size()
            outputs_2d[i] = outputs_2d[i].view(batch_size, channels, dim1, dim3)
            occ2d_pred = self.occ2d[i](outputs_2d[i])
            occ2d_preds.append(occ2d_pred)

        outs = {
            'volume_embed': volume_embed,
            'occ_preds': occ_preds,
            'occ2d_preds':occ2d_preds
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ,
             preds_dicts,
             img_metas):

        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i][:, 0]

                ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)

                # gt = torch.mode(gt, dim=-1)[0].float()

                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(),
                                                                                           semantic=False))

                loss_occ_i = loss_occ_i * ((0.5) ** (len(preds_dicts['occ_preds']) - 1 - i))  # * focal_weight

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        else:
            pred = preds_dicts['occ_preds']
            devicepred = gt_occ.device
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )

            loss_dict = {}

            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i]
                ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)
                pred2d = preds_dicts['occ2d_preds'][i]
                # print("test2")
                # print(preds_dicts['occ_preds'][i].shape)
                # print(gt_occ.shape)
                # print(ratio)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                gt2d = make2dgt(gt_occ.cpu().clone()).clone().to(devicepred)
                gt2d = multiscale_supervision2d(gt2d.clone(), ratio, preds_dicts['occ2d_preds'][i].shape)
                # print("gt2")
                # print(gt.shape)
                # print(gt)
                # print("-----")
                loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred,
                                                                                                          gt.long()))

                loss_occ_i_2d = (criterion(pred2d, gt2d.long()) + sem_scal_loss2d(pred2d, gt2d.long()) + geo_scal_loss2d(pred2d,
                                                                                                          gt2d.long()))
                loss_occ_i = loss_occ_i * ((0.5) ** (len(preds_dicts['occ_preds']) - 1 - i))\
                             +loss_occ_i_2d * ((0.3) ** (len(preds_dicts['occ2d_preds']) - 1 - i))

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict


