#import sys
#print("Python path:", sys.path)
from ..registry import BACKBONE
#print("Registry contents on import:", list(BACKBONE.keys()))

def build_backbone(cfg):
    #print("Available backbones:", list(BACKBONE.keys()))
    assert cfg.MODEL.BACKBONE.TYPE in BACKBONE, \
        "cfg.MODEL.BACKBONE.TYPE: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.TYPE
        )

    if "R" == cfg.MODEL.BACKBONE.TYPE[0]:
        model = BACKBONE[cfg.MODEL.BACKBONE.TYPE](
            pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
            with_ibn=cfg.MODEL.BACKBONE.WITH_IBN
            )
    if "E" == cfg.MODEL.BACKBONE.TYPE[0]:
        model = BACKBONE[cfg.MODEL.BACKBONE.TYPE](
            pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
            )
    
    return model
