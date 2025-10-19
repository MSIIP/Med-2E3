import os
import os.path as osp

import torch
torch.set_float32_matmul_precision('high')  # for speed up
import transformers

from src.data.dataset import create_data_module
from src.model.configuration_medmllm import MedMLLMConfig
from src.model.modeling_medmllm import MedMLLMForConditionalGeneration
from src.trainer_medmllm import MedMLLMTrainer
from src.training_recipe import RECIPE_FACTORY
from src.utils.arguments import ModelArguments, DataArguments, TrainingArguments


from transformers import TrainerCallback
class LrMonitorCallback(TrainerCallback):
    def __init__(self, initial_lr):
        super().__init__()
        self.initial_lr = initial_lr
        self.current_lr = None
        self.lr_ratio = None

    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs['optimizer']
        self.current_lr = optimizer.param_groups[0]['lr']
        self.lr_ratio = self.current_lr / self.initial_lr

        # 将当前学习率传递给模型（如果模型需要监控）
        if 'model' in kwargs and hasattr(kwargs['model'], 'current_lr'):
            kwargs['model'].current_lr = self.current_lr
            kwargs['model'].lr_ratio = self.lr_ratio


def save_args(model_arguments, data_arguments, training_arguments):
    output_dir = osp.join(training_arguments.output_dir, "args")
    os.makedirs(output_dir, exist_ok=True)

    # save model arguments
    with open(osp.join(output_dir, "model_args.txt"), "w") as f:
        f.write(str(model_arguments))
    # save data arguments
    with open(osp.join(output_dir, "data_args.txt"), "w") as f:
        f.write(str(data_arguments))
    # save training arguments
    with open(osp.join(output_dir, "training_args.txt"), "w") as f:
        f.write(str(training_arguments))


def train():
    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load and save args...")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_arguments, data_arguments, training_arguments = (
        parser.parse_args_into_dataclasses()
    )
    save_args(model_arguments, data_arguments, training_arguments)

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Load training_recipe...")
    training_recipe = RECIPE_FACTORY[training_arguments.training_recipe](training_arguments)

    print("*" * 30 + "Stage 3" + "*" * 30)
    print("Load and save model_config...")
    model_config = MedMLLMConfig(model_arguments)
    model_config.save_pretrained(osp.join(training_arguments.output_dir, "args"))

    print("*" * 30 + "Stage 4" + "*" * 30)
    print("Load model...")
    model = MedMLLMForConditionalGeneration(model_config)
    model = training_recipe.load_model(model)
    model = training_recipe(model)

    print("*" * 30 + "Stage 5" + "*" * 30)
    print("Create data_module...")
    data_module = create_data_module(
        model=model,
        data_arguments=data_arguments,
        mode="train",
    )

    print("*" * 30 + "Stage 6" + "*" * 30)
    print("Create trainer and train...")
    lr_monitor = LrMonitorCallback(training_arguments.learning_rate)
    trainer = MedMLLMTrainer(
        model=model,
        tokenizer=model.tokenizer,
        args=training_arguments,
        callbacks=[lr_monitor],
        **data_module,
    )
    trainer.train()

    print("*" * 30 + "Stage 7" + "*" * 30)
    print("Save model...")
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
