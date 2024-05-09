# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer)
from mmcv.runner.base_module import BaseModule, ModuleList

from mmdet.models.utils.builder import TRANSFORMER

from mmdet.models.utils.transformer import Transformer

from .CustomMSDeformableAttention import CustomMSDeformableAttention
from mmdet.models.utils.transformer import inverse_sigmoid
    
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTransformerDecoder_new(BaseModule):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, 
                 transformerlayers=None, 
                 num_layers=None, 
                 prop_add_stage=0,
                 return_intermediate=True,
                 init_cfg=None):
        
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.return_intermediate = return_intermediate
        self.prop_add_stage = prop_add_stage
        assert prop_add_stage >= 0  and prop_add_stage < num_layers

    def forward(self,
                query,
                prop_query,
                key,
                value,
                query_pos,
                key_padding_mask,
                query_key_padding_mask,
                reference_points,
                prop_reference_points,
                spatial_shapes,
                level_start_index,
                reg_branches,
                cls_branches,
                is_first_frame_list,
                predict_refine,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape (bs, num_query, num_points, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        num_queries, bs, embed_dims = query.shape
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None:
                bs, topk, embed_dims = prop_query.shape
                output = output.permute(1, 0, 2)
                with torch.no_grad():
                    tmp_scores, _ = cls_branches[lid](output).max(-1) # (bs, num_q)
                new_query = []
                new_refpts = []
                for i in range(bs):
                    if is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(reference_points[i])
                    else:
                        _, valid_idx = torch.topk(tmp_scores[i], k=num_queries-topk, dim=-1)
                        new_query.append(torch.cat([prop_query[i], output[i][valid_idx]], dim=0))
                        new_refpts.append(torch.cat([prop_reference_points[i], reference_points[i][valid_idx]], dim=0))
                
                output = torch.stack(new_query).permute(1, 0, 2)
                reference_points = torch.stack(new_refpts)
                assert list(output.shape) == [num_queries, bs, embed_dims]

            tmp = reference_points.clone()
            tmp[..., 1:2] = 1.0 - reference_points[..., 1:2] # reverse y-axis
            # reference_points = tmp
            
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=tmp,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_key_padding_mask=None,
                **kwargs)
            
            # reg_points = reg_branches[lid](output.permute(1, 0, 2)) # (bs, num_q, 2*num_points)
            # breakpoint()
            reg_output = reg_branches[lid](output.permute(1, 0, 2))
            reg_points = reg_output[:, :, 0:40] # (bs, num_q, 2*num_points)
            reg_betas = reg_output[:, :, 40:80]
            # breakpoint()
            bs, num_queries, num_points2 = reg_points.shape
            reg_points = reg_points.view(bs, num_queries, num_points2//2, 2) # range (0, 1)
            reg_betas = reg_betas.view(bs, num_queries, num_points2//2, 2) # range (0, 1)
            # breakpoint()
            
            if predict_refine:
                new_reference_points = reg_points + inverse_sigmoid(
                    reference_points
                )
                new_reference_points = new_reference_points.sigmoid()
            else:
                new_reference_points = reg_points.sigmoid() # (bs, num_q, num_points, 2)
            
            reference_points = new_reference_points.clone().detach()
            # reg_betas = reg_betas.clone().detach()
            final_reference_points = torch.cat([reference_points, reg_betas], dim=-1)
            # breakpoint()

            if self.return_intermediate:
                intermediate.append(output.permute(1, 0, 2)) # [(bs, num_q, embed_dims)]
                # intermediate_reference_points.append(new_reference_points) # (bs, num_q, num_points, 2)
                # breakpoint()
                intermediate_reference_points.append(torch.cat([new_reference_points, reg_betas], dim=-1)) # (bs, num_q, num_points, 2)

        if self.return_intermediate:
            # breakpoint()
            return intermediate, intermediate_reference_points

        return output, final_reference_points

@TRANSFORMER_LAYER.register_module()
class MapTransformerLayer(BaseTransformerLayer):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        super().__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            batch_first=batch_first,
            **kwargs
        )

    def forward(self,
                query,
                key=None,
                value=None,
                memory_query=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if memory_query is None:
                    temp_key = temp_value = query
                else:
                    temp_key = temp_value = torch.cat([memory_query, query], dim=0)
                
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER.register_module()
class MapTransformer(Transformer):
    """Implements the DeformableDETR transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=1,
                 num_points=20,
                 coord_dim=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_feature_levels = num_feature_levels
        self.embed_dims = self.encoder.embed_dims
        self.coord_dim = coord_dim
        self.num_points = num_points
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, CustomMSDeformableAttention):
                m.init_weights()

        # normal_(self.level_embeds)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                init_reference_points,
                reg_branches=None,
                cls_branches=None,
                memory_query=None,
                prop_query=None,
                prop_reference_points=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        feat_flatten = []
        mask_flatten = []
        # lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            # pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            # lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        # decoder
        query = query_embed.permute(1, 0, 2) # (num_q, bs, embed_dims)
        if memory_query is not None:
            memory_query = memory_query.permute(1, 0, 2)
        # breakpoint()
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=feat_flatten,
            query_pos=None,
            key_padding_mask=mask_flatten,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            memory_query=memory_query,
            prop_query=prop_query,
            prop_reference_points=prop_reference_points,
            **kwargs)

        return inter_states, init_reference_points, inter_references