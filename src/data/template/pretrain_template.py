import copy
from dataclasses import dataclass

from src.data.template.base import Template
from src.data.template.formatter import Formatter, EmptyFormatter, StringFormatter
from src.utils.constants import IGNORE_INDEX


@dataclass
class PretrainTemplate(Template):
    system: "Formatter" = EmptyFormatter(slot="")
    format_image_token: "Formatter" = EmptyFormatter(slot="")
    format_user: "Formatter" = EmptyFormatter(slot="<image>")
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}")
    separator: "Formatter" = EmptyFormatter(slot=["", ""])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        instruction_len = len(self.tokenizer_special_token("<image>", tokenizer))
        labels[:instruction_len] = IGNORE_INDEX
        return labels
