# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from networks.PoolNets import PoolFormer
from networks.ResNets import ResNet,Bottleneck
from networks.swin_transformer import SwinTransformer
from networks.swin_mlp import SwinMLP
from networks.swin_transformer_v2 import SwinTransformerV2
from networks.mobilenetv2 import mobilenet_v2
from networks.models_config import cfg_dict

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


def build_backbone(config):
    model_type = config.MODEL.TYPE
    
    pretrained = config.MODEL.PRETRAINED

    embed_dims = []
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
        
        embed_dims = [config.MODEL.SWIN.EMBED_DIM*2**i for i in range(4)]
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
        embed_dims = [config.MODEL.SWINV2.EMBED_DIM*2**i for i in range(4)]
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        embed_dims = [config.MODEL.SWIN_MLP.EMBED_DIM*2**i for i in range(4)]
    elif model_type == "pool":
        model = PoolFormer(config.MODEL.POOL.LAYERS, embed_dims=config.MODEL.POOL.EMBED_DIMS, 
            mlp_ratios=config.MODEL.POOL.MLP_RATIOS, downsamples=config.MODEL.POOL.DOWNSAMPLES, 
            layer_scale_init_value=1e-6, 
            fork_feat=True)
        model.default_cfg = default_cfgs[config.MODEL.POOL.TYPE]
        embed_dims = config.MODEL.POOL.EMBED_DIMS
    elif model_type == "resnet":
        model_type = config.MODEL.TYPE
        pretrained = config.MODEL.PRETRAINED
        model = ResNet(Bottleneck, config.MODEL.RESNET.LAYERS)
        if pretrained:
            ckpt_path = cfg_dict[config.MODEL.NICKNAME][0]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt,strict=False)

        return model,config.MODEL.RESNET.EMBED_DIMS
    elif model_type == "mobilenet":
        model_type = config.MODEL.TYPE
        pretrained = config.MODEL.PRETRAINED
        model = mobilenet_v2(False)
        if pretrained:
            ckpt_path = cfg_dict[config.MODEL.NICKNAME][0]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt,strict=False)

        return model,config.MODEL.MOBILE.EMBED_DIMS
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    if pretrained:
        ckpt_path = cfg_dict[config.MODEL.NICKNAME][0]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'],strict=False)
    return model,embed_dims


if __name__ == "__main__":
    pass