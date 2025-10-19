from src.model.encoder.vision2d.clip import CLIPVision2DModel, BiomedCLIPVision2DModel
from src.model.encoder.vision2d.siglip import SiglipVision2DModel
from src.model.encoder.vision2d.dino import Dinov2Vision2DModel

VISION2D_FACTORY = {
    "clip": CLIPVision2DModel,
    "siglip": SiglipVision2DModel,
    "dinov2": Dinov2Vision2DModel,
    "biomedclip": BiomedCLIPVision2DModel,
}
