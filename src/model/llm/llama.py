from transformers import LlamaForCausalLM, AutoTokenizer


def return_llamaclass():
    def post_load(tokenizer):
        return tokenizer

    return (LlamaForCausalLM, (AutoTokenizer, post_load))


def return_llama3class():
    def post_load(tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    return (LlamaForCausalLM, (AutoTokenizer, post_load))
