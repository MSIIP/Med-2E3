from src.model.llm.llama import return_llamaclass, return_llama3class
from src.model.llm.phi import return_phiclass, return_phi3class
from src.model.llm.qwen import return_qwen2class

LLM_FACTORY = {
    "llama": return_llamaclass(),
    "llama3": return_llama3class(),
    "phi": return_phiclass(),
    "phi3": return_phi3class(),
    "qwen2": return_qwen2class(),
}
