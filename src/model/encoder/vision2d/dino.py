from transformers import AutoImageProcessor, Dinov2Model

from src.model.encoder.vision2d.base import Vision2DModel


class Dinov2Vision2DModel(Vision2DModel):
    def __init__(self, vision2d_model_config, cache_dir_hf=None):
        super().__init__(vision2d_model_config, cache_dir_hf)

        self.vision2d_processor = AutoImageProcessor.from_pretrained(
            vision2d_model_config._name_or_path,
            cache_dir=self.cache_dir_hf,
            trust_remote_code=True,
        )
        self.vision2d_encoder = Dinov2Model(vision2d_model_config)