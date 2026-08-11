import torch
from torch import nn

import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.layers.wrappers import Conv2d 
from detectron2.layers.batch_norm import get_norm 

from .base import DataFusionBlock
from .build import DATAFUSION_REGISTRY


class OpticalFlowFusionBlock(DataFusionBlock):
    """
    OpticalFlowFusionBlock takes 2 features. The first is the raw 3 channel image data,
    the second is the 3 channel optical flow feature extracted from that image data.
    Each is projected by its own 1x1 convolution, the two are concatenated, and the
    result is projected back to 3 channels so the fused feature can be fed straight
    into the backbone stem without changing its input width.
    """

    def __init__(self, 
                 raw_image_in_channels: int = 3,
                 raw_image_out_channels: int = 8,
                 optical_flow_in_channels: int = 3,
                 optical_flow_out_channels: int = 8,
                 fused_feature_out_channels: int = 3,
                 norm: str = "BN"):
        
        super().__init__()

        self.conv1 = Conv2d(
            raw_image_in_channels,
            raw_image_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm(norm, raw_image_out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

        self.conv2 = Conv2d(
            optical_flow_in_channels,
            optical_flow_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm(norm, optical_flow_out_channels),
        )
        weight_init.c2_msra_fill(self.conv2)

        fusion_conv_in_channel = raw_image_out_channels + optical_flow_out_channels
        self.conv3 = Conv2d(
            fusion_conv_in_channel,
            fused_feature_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm(norm, fused_feature_out_channels),
        )
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, 
                raw_image: torch.Tensor, 
                optical_flow_image: torch.Tensor):
        
        raw_image_feature = self.conv1(raw_image)
        raw_image_feature = F.relu_(raw_image_feature)

        optical_flow_image_feature = self.conv2(optical_flow_image)
        optical_flow_image_feature = F.relu_(optical_flow_image_feature)

        concatenated_feature = torch.cat((raw_image_feature, optical_flow_image_feature), dim=1)
        concatenated_feature = self.conv3(concatenated_feature)
        fused_feature = F.relu_(concatenated_feature)

        return fused_feature


@DATAFUSION_REGISTRY.register()
def build_optical_flow_fusion_block(cfg):
    """
    Reads parameters of optical flow data fusion block from cfg.

    Return
        OpticalFlowFusionBlock which is used to feed backbone.
    """
    raw_image_in_channels           = cfg.MODEL.DATAFUSION.RAW_IMAGE.IN_FEATURES
    raw_image_out_channels          = cfg.MODEL.DATAFUSION.RAW_IMAGE.OUT_FEATURES
    optical_flow_in_channels        = cfg.MODEL.DATAFUSION.OPTICAL_FLOW.IN_FEATURES
    optical_flow_out_channels       = cfg.MODEL.DATAFUSION.OPTICAL_FLOW.OUT_FEATURES
    fused_feature_out_channels      = cfg.MODEL.DATAFUSION.OUT_FEATURES
    norm                            = cfg.MODEL.DATAFUSION.NORM
    
    return OpticalFlowFusionBlock(raw_image_in_channels=raw_image_in_channels,
                                  raw_image_out_channels=raw_image_out_channels,
                                  optical_flow_in_channels=optical_flow_in_channels,
                                  optical_flow_out_channels=optical_flow_out_channels,
                                  fused_feature_out_channels=fused_feature_out_channels,
                                  norm=norm)