import os.path as osp

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class Vision3DModel(nn.Module):
    def __init__(self, vision3d_model_config, cache_dir_hf=None):
        super().__init__()
        self.cache_dir_hf = cache_dir_hf
        self.vision3d_model_config = vision3d_model_config
        self.vision3d_processor = None
        self.vision3d_encoder = None

    def load_model(self, vision3d_model_name_or_path):
        if isinstance(self.vision3d_encoder, PreTrainedModel):
            self.vision3d_encoder = self.vision3d_encoder.from_pretrained(
                vision3d_model_name_or_path,
                cache_dir=self.cache_dir_hf,
            )
        else:
            vision3d_encoder_weights = torch.load(
                osp.join(vision3d_model_name_or_path, "pytorch_model.bin"),
                map_location="cpu",
            )
            self.vision3d_encoder.load_state_dict(vision3d_encoder_weights)

        self.vision3d_encoder.requires_grad_(False)

    def forward(self, x, vision3d_model_select_layer, vision3d_model_select_feature):
        vision3d_features = self.vision3d_encoder(x, output_hidden_states=True)
        vision3d_features = vision3d_features.hidden_states[vision3d_model_select_layer]

        if vision3d_model_select_feature == "patch":
            vision3d_features = vision3d_features[:, 1:]
        elif vision3d_model_select_feature == "cls_patch":
            vision3d_features = vision3d_features

        return vision3d_features
