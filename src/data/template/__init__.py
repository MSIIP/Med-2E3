from src.data.template.pretrain_template import PretrainTemplate
from src.data.template.pretrain_vision3d_template import PretrainVision3DTemplate
from src.data.template.llama_template import LlamaTemplate, Llama3Template
from src.data.template.phi_template import PhiTemplate
from src.data.template.qwen_template import Qwen2Template

TEMPlATE_FACTORY = {
    "pretrain": PretrainTemplate,
    "pretrain_vision3d": PretrainVision3DTemplate,
    "llama": LlamaTemplate,
    "llama3": Llama3Template,
    "phi": PhiTemplate,
    "qwen2": Qwen2Template,
}
