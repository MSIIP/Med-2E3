from transformers import Qwen2ForCausalLM, AutoTokenizer


def return_qwen2class():
    def post_load(tokenizer):
        return tokenizer

    return (Qwen2ForCausalLM, (AutoTokenizer, post_load))
