from transformers import AutoModel, PreTrainedModel

from src.model.encoder.vision3d.base import Vision3DModel


class M3DVision3DModel(Vision3DModel):
    def __init__(self, vision3d_model_config, cache_dir_hf=None):
        super().__init__(vision3d_model_config, cache_dir_hf)
        self.vision3d_encoder = AutoModel.from_pretrained(
            vision3d_model_config._name_or_path,
            cache_dir=self.cache_dir_hf,
            trust_remote_code=True,
        )

    def load_model(self, vision3d_model_name_or_path):
        if isinstance(self.vision3d_encoder, PreTrainedModel):
            self.vision3d_encoder = self.vision3d_encoder.from_pretrained(
                vision3d_model_name_or_path,
                cache_dir=self.cache_dir_hf,
            )
        self.vision3d_encoder.requires_grad_(False)

    def forward(self, x, vision3d_model_select_layer, vision3d_model_select_feature):
        _, vision3d_features = self.vision3d_encoder.vision_encoder(x)
        vision3d_features = vision3d_features[vision3d_model_select_layer]

        if vision3d_model_select_feature == "patch":
            vision3d_features = vision3d_features[:, 1:]
        elif vision3d_model_select_feature == "cls_patch":
            vision3d_features = vision3d_features

        return vision3d_features
