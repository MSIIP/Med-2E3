import os
import os.path as osp

import torch
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from src.training_recipe.base import BaseTrainingRecipe
from src.utils.train_utils import (
    find_all_linear_names,
    get_peft_state_non_lora_maybe_zero_3,
    get_peft_state_maybe_zero_3,
)


class LoRATrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.lora_skip_module = [
            "llm",
            "vision2d_model",
            "vision2d_connector",
            "vision3d_model",
            "vision3d_connector",
        ]

    def training_model_converse(self, model):
        if self.training_arguments.tune_type_llm == "lora":
            self.lora_skip_module.remove("llm")
        if self.training_arguments.tune_type_vision2d_model == "lora":
            self.lora_skip_module.remove("vision2d_model")
        if self.training_arguments.tune_type_vision2d_connector == "lora":
            self.lora_skip_module.remove("vision2d_connector")
        if self.training_arguments.tune_type_vision3d_model == "lora":
            self.lora_skip_module.remove("vision3d_model")
        if self.training_arguments.tune_type_vision3d_connector == "lora":
            self.lora_skip_module.remove("vision3d_connector")

        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            target_modules=find_all_linear_names(model, self.lora_skip_module),
            task_type="CAUSAL_LM",
        )

        # TODO: 16bit

        model = get_peft_model(model, lora_config)
        return model

    def save(self, model, trainer):
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        trainer.save_state()

        if trainer.deepspeed:
            torch.cuda.synchronize()

        # save llm base params
        llm_state_dict = get_peft_state_non_lora_maybe_zero_3(model.llm.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            llm_output_dir = osp.join(self.training_arguments.output_dir, "llm")
            os.makedirs(llm_output_dir, exist_ok=True)
            llm_output_path = osp.join(self.training_arguments.output_dir, "llm/pytorch_model.bin")
            torch.save(llm_state_dict, llm_output_path)
            model.config.llm_config.save_pretrained(llm_output_dir, from_pt=True)

        # save vision2d model base params
        vision2d_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision2d_model.vision2d_encoder.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision2d_model_output_dir = osp.join(self.training_arguments.output_dir, "vision2d_model")
            os.makedirs(vision2d_model_output_dir, exist_ok=True)
            vision2d_model_output_path = osp.join(self.training_arguments.output_dir, "vision2d_model/pytorch_model.bin")
            torch.save(vision2d_model_state_dict, vision2d_model_output_path)
            # TODO: TinyLLaVA保存的是model.config.vision2d_model_config?
            if isinstance(model.vision2d_model.vision2d_encoder, PreTrainedModel):
                model.vision2d_model.vision2d_encoder.config.save_pretrained(vision2d_model_output_dir, from_pt=True)

        # save vision2d connector base params
        vision2d_connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision2d_connector.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision2d_connector_output_dir = osp.join(self.training_arguments.output_dir, "vision2d_connector")
            os.makedirs(vision2d_connector_output_dir, exist_ok=True)
            vision2d_connector_output_path = osp.join(self.training_arguments.output_dir, "vision2d_connector/pytorch_model.bin")
            torch.save(vision2d_connector_state_dict, vision2d_connector_output_path)

        # save vision3d model base params
        vision3d_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision3d_model.vision3d_encoder.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision3d_model_output_dir = osp.join(self.training_arguments.output_dir, "vision3d_model")
            os.makedirs(vision3d_model_output_dir, exist_ok=True)
            vision3d_model_output_path = osp.join(self.training_arguments.output_dir, "vision3d_model/pytorch_model.bin")
            torch.save(vision3d_model_state_dict, vision3d_model_output_path)
            if isinstance(model.vision3d_model.vision3d_encoder, PreTrainedModel):
                model.vision3d_model.vision3d_encoder.config.save_pretrained(vision3d_model_output_dir, from_pt=True)

        # save vision3d connector base params
        vision3d_connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision3d_connector.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision3d_connector_output_dir = osp.join(self.training_arguments.output_dir, "vision3d_connector")
            os.makedirs(vision3d_connector_output_dir, exist_ok=True)
            vision3d_connector_output_path = osp.join(self.training_arguments.output_dir, "vision3d_connector/pytorch_model.bin")
            torch.save(vision3d_connector_state_dict, vision3d_connector_output_path)

        # save lora params
        lora_state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), self.training_arguments.lora_bias)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
