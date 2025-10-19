import os.path as osp
import torch
from open_clip import create_model_from_pretrained
from transformers import CLIPImageProcessor, CLIPVisionModel, AutoModel

from src.model.encoder.vision2d.base import Vision2DModel


class CLIPVision2DModel(Vision2DModel):
    def __init__(self, vision2d_model_config, cache_dir_hf=None):
        super().__init__(vision2d_model_config, cache_dir_hf)

        self.vision2d_processor = CLIPImageProcessor.from_pretrained(
            vision2d_model_config._name_or_path,
            cache_dir=self.cache_dir_hf,
        )
        self.vision2d_encoder = CLIPVisionModel(vision2d_model_config)


class BiomedCLIPVision2DModel(Vision2DModel):
    def __init__(self, vision2d_model_config, cache_dir_hf=None):
        super().__init__(vision2d_model_config, cache_dir_hf)

        # model, preprocess = create_model_from_pretrained(
        #     'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        #     cache_dir=cache_dir_hf,
        # )
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True, cache_dir=cache_dir_hf)
        self.vision2d_processor = None
        self.vision2d_encoder = model.vision_model

    def load_model(self, vision2d_model_name_or_path):
        self.vision2d_encoder = AutoModel.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            cache_dir=self.cache_dir_hf,
        )
        if hasattr(self.vision2d_encoder, "vision_model"):
            self.vision2d_encoder = getattr(self.vision2d_encoder, "vision_model", self.vision2d_encoder)
        # vision2d_encoder_weights = torch.load(
        #     osp.join(vision2d_model_name_or_path, "pytorch_model.bin"),
        #     map_location="cpu",
        # )
        # self.vision2d_encoder.load_state_dict(vision2d_encoder_weights)

        self.vision2d_encoder.requires_grad_(False)

    def forward(self, x, vision2d_model_select_layer, vision2d_model_select_feature):
        vision2d_features = self.vision2d_encoder(x, output_hidden_states=True)
        vision2d_features = vision2d_features.hidden_states[vision2d_model_select_layer]

        if vision2d_model_select_feature == "patch":
            vision2d_features = vision2d_features[:, 1:]
        elif vision2d_model_select_feature == "cls_patch":
            vision2d_features = vision2d_features

        return vision2d_features