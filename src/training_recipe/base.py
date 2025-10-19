import os
import os.path as osp

import torch
from transformers import PreTrainedModel

from src.utils.train_utils import get_state_maybe_zero_3


class BaseTrainingRecipe:
    def __init__(self, training_arguments):
        self.training_arguments = training_arguments

    def load_model(self, model):
        if self.training_arguments.resume_from_checkpoint is not None:
            llm_path = osp.join(self.training_arguments.resume_from_checkpoint, "llm")
            vision2d_model_path = osp.join(self.training_arguments.resume_from_checkpoint, "vision2d_model")
            vision2d_connector_path = osp.join(self.training_arguments.resume_from_checkpoint, "vision2d_connector")
            vision3d_model_path = osp.join(self.training_arguments.resume_from_checkpoint, "vision3d_model")
            vision3d_connector_path = osp.join(self.training_arguments.resume_from_checkpoint, "vision3d_connector")
        else:
            llm_path = None
            vision2d_model_path = None
            vision2d_connector_path = None
            vision3d_model_path = None
            vision3d_connector_path = None

        model.load_vision2d_model(vision2d_model_path)
        model.load_vision2d_connector(vision2d_connector_path)
        model.load_vision3d_model(vision3d_model_path)
        model.load_vision3d_connector(vision3d_connector_path)

        if (self.training_arguments.resume_from_checkpoint is not None) and (osp.exists(osp.join(self.training_arguments.resume_from_checkpoint, "adapter_config.json"))):
            model.llm = model.llm.from_pretrained(
                pretrained_model_name_or_path=model.config.llm_config._name_or_path,
                torch_dtype=torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32),
                attn_implementation=model.config.llm_attn_implementation,
                cache_dir=model.config.cache_dir_hf,
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.training_arguments.resume_from_checkpoint)
            model = model.merge_and_unload()
        else:
            model.load_llm(
                llm_path,
                torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32),
                model.config.llm_attn_implementation,
            )

        return model

    def __call__(self, model):
        model = self.training_model_converse(model)
        model = self.tune_type_setting(model)

        # `use_cache=True` is incompatible with gradient checkpointing.
        if self.training_arguments.gradient_checkpointing:
            model.config.use_cache = False

        return model

    def training_model_converse(self, model):
        return model

    def tune_type_setting(self, model):
        model = self._llm_tune_type_setting(model)
        model = self._vision2d_model_tune_type_setting(model)
        model = self._vision2d_connector_tune_type_setting(model)
        model = self._vision3d_model_tune_type_setting(model)
        model = self._vision3d_connector_tune_type_setting(model)
        return model

    def _llm_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_llm.lower()
        if tune_type == "full":
            model.llm.requires_grad_(True)
        elif tune_type == "frozen":
            model.llm.requires_grad_(False)

        # gradient checkpointing是一种优化训练深度神经网络时内存占用的技术
        self.support_gradient_checkpoint(model.llm, self.training_arguments.gradient_checkpointing)

        return model

    def _vision2d_model_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision2d_model.lower()
        if tune_type == "full":
            model.vision2d_model.requires_grad_(True)
        elif tune_type == "frozen":
            model.vision2d_model.requires_grad_(False)
        return model

    def _vision2d_connector_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision2d_connector.lower()
        if tune_type == "full":
            for p in model.vision2d_connector.parameters():
                p.requires_grad = True
        elif tune_type == "frozen":
            for p in model.vision2d_connector.parameters():
                p.requires_grad = False
        return model

    def _vision3d_model_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision3d_model.lower()
        if tune_type == "full":
            model.vision3d_model.requires_grad_(True)
        elif tune_type == "frozen":
            model.vision3d_model.requires_grad_(False)
        return model

    def _vision3d_connector_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision3d_connector.lower()
        if tune_type == "full":
            for p in model.vision3d_connector.parameters():
                p.requires_grad = True
        elif tune_type == "frozen":
            for p in model.vision3d_connector.parameters():
                p.requires_grad = False
        return model

    def support_gradient_checkpoint(self, model, gradient_checkpointing=False):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def save(self, model, trainer):
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        trainer.save_state()

        if trainer.deepspeed:
            torch.cuda.synchronize()

        # save medmllm
        # trainer.save_model(osp.join(self.training_arguments.output_dir, "mllm"))

        # save llm
        llm_state_dict = get_state_maybe_zero_3(model.llm.named_parameters(), [""], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            llm_output_dir = osp.join(self.training_arguments.output_dir, "llm")
            os.makedirs(llm_output_dir, exist_ok=True)
            llm_output_path = osp.join(self.training_arguments.output_dir, "llm/pytorch_model.bin")
            torch.save(llm_state_dict, llm_output_path)
            model.config.llm_config.save_pretrained(llm_output_dir, from_pt=True)

        # save vision2d model
        vision2d_model_state_dict = get_state_maybe_zero_3(model.vision2d_model.vision2d_encoder.named_parameters(), [""], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision2d_model_output_dir = osp.join(self.training_arguments.output_dir, "vision2d_model")
            os.makedirs(vision2d_model_output_dir, exist_ok=True)
            vision2d_model_output_path = osp.join(self.training_arguments.output_dir, "vision2d_model/pytorch_model.bin")
            torch.save(vision2d_model_state_dict, vision2d_model_output_path)
            if isinstance(model.vision2d_model.vision2d_encoder, PreTrainedModel):
                model.vision2d_model.vision2d_encoder.config.save_pretrained(vision2d_model_output_dir, from_pt=True)

        # save vision2d connector
        vision2d_connector_state_dict = get_state_maybe_zero_3(model.vision2d_connector.named_parameters(), [""], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision2d_connector_output_dir = osp.join(self.training_arguments.output_dir, "vision2d_connector")
            os.makedirs(vision2d_connector_output_dir, exist_ok=True)
            vision2d_connector_output_path = osp.join(self.training_arguments.output_dir, "vision2d_connector/pytorch_model.bin")
            torch.save(vision2d_connector_state_dict, vision2d_connector_output_path)

        # save vision3d model
        vision3d_model_state_dict = get_state_maybe_zero_3(model.vision3d_model.vision3d_encoder.named_parameters(), [""], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision3d_model_output_dir = osp.join(self.training_arguments.output_dir, "vision3d_model")
            os.makedirs(vision3d_model_output_dir, exist_ok=True)
            vision3d_model_output_path = osp.join(self.training_arguments.output_dir, "vision3d_model/pytorch_model.bin")
            torch.save(vision3d_model_state_dict, vision3d_model_output_path)
            if isinstance(model.vision3d_model.vision3d_encoder, PreTrainedModel):
                model.vision3d_model.vision3d_encoder.config.save_pretrained(vision3d_model_output_dir, from_pt=True)

        # save vision3d connector
        vision3d_connector_state_dict = get_state_maybe_zero_3(model.vision3d_connector.named_parameters(), [""], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision3d_connector_output_dir = osp.join(self.training_arguments.output_dir, "vision3d_connector")
            os.makedirs(vision3d_connector_output_dir, exist_ok=True)
            vision3d_connector_output_path = osp.join(self.training_arguments.output_dir, "vision3d_connector/pytorch_model.bin")
            torch.save(vision3d_connector_state_dict, vision3d_connector_output_path)