from .base import DataFusionBlock

from detectron2.utils.registry import Registry

DATAFUSION_REGISTRY = Registry("DATAFUSION")
DATAFUSION_REGISTRY.__doc__ = """
Registry for data fusions, which fusion raw images and additional features

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_optical_flow_fusion_block(cfg):
    """
    Build the data fusion block named by `cfg.MODEL.DATAFUSION.NAME`.
    Returns None when `cfg.MODEL.DATAFUSION.STATUS` is false, which leaves
    the model on the stock MaskFreeVIS path.
    """
    data_fusion_status = cfg.MODEL.DATAFUSION.STATUS
    
    if data_fusion_status:
        data_fusion_block_name = cfg.MODEL.DATAFUSION.NAME
        data_fusion_block = DATAFUSION_REGISTRY.get(data_fusion_block_name)(cfg)
        assert isinstance(data_fusion_block, DataFusionBlock)
        return data_fusion_block

    return None