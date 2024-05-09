import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.resnet import resnet18, resnet50

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck)

from .base_mapper import BaseMapper, MAPPERS
from copy import deepcopy
from ..utils.memory_buffer import StreamTensorMemory
from mmcv.cnn.utils import constant_init, kaiming_init

@MAPPERS.register_module()
class StreamMapNet(BaseMapper):

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 head_cfg=dict(),
                 neck_cfg=None,
                 model_name=None, 
                 streaming_cfg=dict(),
                 pretrained=None,
                 **kwargs):
        super().__init__()

        #Attribute
        self.model_name = model_name
        self.last_epoch = None
  
        self.backbone = build_backbone(backbone_cfg)

        if neck_cfg is not None:
            self.neck = build_head(neck_cfg)
        else:
            self.neck = nn.Identity()

        self.head = build_head(head_cfg)
        self.num_decoder_layers = self.head.transformer.decoder.num_layers
        
        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size

        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
        else:
            self.streaming_bev = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])
            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )
            
            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
        
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            try:
                self.neck.init_weights()
            except AttributeError:
                pass
            if self.streaming_bev:
                self.stream_fusion_neck.init_weights()

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                # else, warp buffered bev feature to current pose
                prev_e2g_trans = self.plane.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.plane.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.plane.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.plane.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_g2e_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                prev_g2e_matrix[:3, :3] = prev_e2g_rot.T
                prev_g2e_matrix[:3, 3] = -(prev_e2g_rot.T @ prev_e2g_trans)

                curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                curr_e2g_matrix[:3, :3] = curr_e2g_rot
                curr_e2g_matrix[:3, 3] = curr_e2g_trans

                curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix
                prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                # from (-30, 30) or (-15, 15) to (-1, 1)
                prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                prev_coord[..., 1] = -prev_coord[..., 1] / (self.roi_size[1]/2)

                warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0), 
                                prev_coord.unsqueeze(0), 
                                padding_mode='zeros', align_corners=False).squeeze(0)
                new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        
        return fused_feats

    def forward_train(self, img, vectors, points=None, img_metas=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images

        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        
        bs = img.shape[0]

        # Backbone
        _bev_feats = self.backbone(img, img_metas=img_metas, points=points)
        
        if self.streaming_bev:
            self.bev_memory.train()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
        
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list, loss_dict, det_match_idxs, det_match_gt_idxs = self.head(
            bev_features=bev_feats, 
            img_metas=img_metas, 
            gts=gts,
            return_loss=True)
        
        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        _bev_feats = self.backbone(img, img_metas, points=points)
        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]

        if self.streaming_bev:
            self.bev_memory.eval()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
            
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
        
        # take predictions from the last layer
        preds_dict = preds_list[-1]

        results_list = self.head.post_process(preds_dict, tokens)

        return results_list

    def batch_data(self, vectors, imgs, img_metas, device, points=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        assert len(valid_idx) == bs # make sure every sample has gts

        gts = []
        all_labels_list = []
        all_lines_list = []
        for idx in range(bs):
            labels = []
            lines = []
            for label, _lines in vectors[idx].items():
                for _line in _lines:
                    labels.append(label)
                    if len(_line.shape) == 3: # permutation
                        num_permute, num_points, coords_dim = _line.shape
                        lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
                    elif len(_line.shape) == 2:
                        lines.append(torch.tensor(_line).reshape(-1)) # (40, )
                    else:
                        assert False

            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            all_lines_list.append(torch.stack(lines).float().to(device))

        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list
        }
        
        gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]

        return gts, imgs, img_metas, valid_idx, points

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()

