import os.path as osp

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class Vision2DModel(nn.Module):
    def __init__(self, vision2d_model_config, cache_dir_hf=None):
        super().__init__()
        self.cache_dir_hf = cache_dir_hf
        self.vision2d_model_config = vision2d_model_config
        self.vision2d_processor = None
        self.vision2d_encoder = None

    def load_model(self, vision2d_model_name_or_path):
        if isinstance(self.vision2d_encoder, PreTrainedModel):
            self.vision2d_encoder = self.vision2d_encoder.from_pretrained(
                vision2d_model_name_or_path,
                cache_dir=self.cache_dir_hf,
            )
        else:
            vision2d_encoder_weights = torch.load(
                osp.join(vision2d_model_name_or_path, "pytorch_model.bin"),
                map_location="cpu",
            )
            self.vision2d_encoder.load_state_dict(vision2d_encoder_weights)

        self.vision2d_encoder.requires_grad_(False)

    def forward(self, x, vision2d_model_select_layer, vision2d_model_select_feature):
        vision2d_features = self.vision2d_encoder(x, output_hidden_states=True)
        vision2d_features = vision2d_features.hidden_states[vision2d_model_select_layer]

        if vision2d_model_select_feature == "patch":
            vision2d_features = vision2d_features[:, 1:]
        elif vision2d_model_select_feature == "cls_patch":
            vision2d_features = vision2d_features

        return vision2d_features
