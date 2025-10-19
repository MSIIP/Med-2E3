from dataclasses import dataclass

from src.data.template.base import Template
from src.data.template.formatter import Formatter, EmptyFormatter, StringFormatter


system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@dataclass
class Qwen2Template(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_image_token: "Formatter" = StringFormatter(slot="<image>{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "<|im_end|>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "<|im_end|>"])
