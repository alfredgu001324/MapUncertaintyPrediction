import copy
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from .bevformer.grid_mask import GridMask
from mmdet3d.models import builder


class UpsampleBlock(nn.Module):
    def __init__(self, ins, outs):
        super(UpsampleBlock, self).__init__()
        self.gn = nn.GroupNorm(32, outs)
        self.conv = nn.Conv2d(ins, outs, kernel_size=3,
                              stride=1, padding=1)  # same
        self.relu = nn.ReLU(inplace=True)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(self.gn(x))
        x = self.upsample2x(x)

        return x

    def upsample2x(self, x):
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(h*2, w*2),
                          mode='bilinear', align_corners=True)
        return x

@BACKBONES.register_module()
class BEVFormerBackbone(nn.Module):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 roi_size,
                 bev_h,
                 bev_w,
                 img_backbone=None, 
                 img_neck=None,               
                 transformer=None,
                 positional_encoding=None,
                 use_grid_mask=True,
                 upsample=False,
                 up_outdim=128,
                 **kwargs):
        super(BEVFormerBackbone, self).__init__()

        # image feature
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.real_w = roi_size[0]
        self.real_h = roi_size[1]
    
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        
        self.upsample = upsample
        if self.upsample:
            self.up = UpsampleBlock(self.transformer.embed_dims, up_outdim)

        self._init_layers()
        self.init_weights()


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        self.img_backbone.init_weights()
        self.img_neck.init_weights()
       
        if self.upsample:
            self.up.init_weights()
    
    # @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped

    def forward(self, img, img_metas, *args, prev_bev=None, only_bev=False, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                            device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outs =  self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        
        outs = outs.unflatten(1,(self.bev_h,self.bev_w)).permute(0,3,1,2).contiguous()
        
        if self.upsample:
            outs = self.up(outs)
        
        return outs
