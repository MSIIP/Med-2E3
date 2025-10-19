from src.training_recipe.base import BaseTrainingRecipe
from src.training_recipe.lora import LoRATrainingRecipe

RECIPE_FACTORY = {
    "common": BaseTrainingRecipe,
    "lora": LoRATrainingRecipe,
}
