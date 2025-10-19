from transformers import SiglipImageProcessor, SiglipVisionModel

from src.model.encoder.vision2d.base import Vision2DModel


class SiglipVision2DModel(Vision2DModel):
    def __init__(self, vision2d_model_config, cache_dir_hf=None):
        super().__init__(vision2d_model_config, cache_dir_hf)

        self.vision2d_processor = SiglipImageProcessor.from_pretrained(
            vision2d_model_config._name_or_path,
            cache_dir=self.cache_dir_hf,
        )
        self.vision2d_encoder = SiglipVisionModel(vision2d_model_config)

    # def forward(self, x, vision2d_model_select_layer, vision2d_model_select_feature):
    #     vision2d_features = self.vision2d_encoder(x, output_hidden_states=True)
    #     pooler_output = vision2d_features.pooler_output
    #     vision2d_features = vision2d_features.hidden_states[vision2d_model_select_layer]

    #     if vision2d_model_select_feature == "patch":
    #         vision2d_features = vision2d_features[:, 1:]
    #     elif vision2d_model_select_feature == "cls_patch":
    #         vision2d_features = vision2d_features

    #     return vision2d_features, pooler_output