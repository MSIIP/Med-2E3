from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    cache_dir_hf: Optional[str] = field(default=None)
    exp_id: Optional[int] = field(default=0)

    # llm
    tokenizer_name_or_path: Optional[str] = field(default=None)
    # 虽然TokenizerFast比普通的tokenizer处理速度快很多，但可能会导致decode后与原文不一致
    tokenizer_use_fast: Optional[bool] = field(default=False)
    llm_name_or_path: Optional[str] = field(default="microsoft/phi-2")
    llm_type: Optional[str] = field(default="phi")
    llm_max_length: Optional[int] = field(default=512)
    llm_padding_side: Optional[str] = field(default="right")
    llm_attn_implementation: Optional[str] = field(default=None)

    # vision2d
    vision2d_model_name_or_path: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    vision2d_model_type: Optional[str] = field(default="clip")
    vision2d_model_select_layer: Optional[int] = field(default=-1)
    vision2d_model_select_feature: Optional[str] = field(default="patch")
    vision2d_connector_type: Optional[str] = field(default="mlp2x_gelu")

    # vision3d
    vision3d_model_name_or_path: Optional[str] = field(default="GoodBaiBai88/M3D-CLIP")
    vision3d_model_type: Optional[str] = field(default="m3d")
    vision3d_model_select_layer: Optional[int] = field(default=-1)
    vision3d_model_select_feature: Optional[str] = field(default="patch")
    vision3d_connector_type: Optional[str] = field(default="spatial_pooling_mlp2x_gelu")

    # 设置print格式，每个属性打印一行，开头打印类名
    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(f"{k}={v}" for k, v in self.__dict__.items())
            + ",\n)"
        )


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    conv_version: str = field(default="pretrain")

    vision2d_data_path: Optional[str] = field(default=None)
    vision3d_data_path: Optional[str] = field(default=None)

    # 设置print格式，每个属性打印一行，开头打印类名
    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(f"{k}={v}" for k, v in self.__dict__.items())
            + ",\n)"
        )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default="common")
    tune_type_llm: str = field(default="frozen")
    tune_type_vision2d_model: str = field(default="frozen")
    tune_type_vision2d_connector: str = field(default="full")
    tune_type_vision3d_model: str = field(default="frozen")
    tune_type_vision3d_connector: str = field(default="full")

    # lora
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_bias: Optional[str] = field(default="none")
