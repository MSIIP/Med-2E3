from transformers import PhiForCausalLM, Phi3ForCausalLM, AutoTokenizer


def return_phiclass():
    def post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        return tokenizer

    return (PhiForCausalLM, (AutoTokenizer, post_load))


def return_phi3class():
    def post_load(tokenizer):
        return tokenizer

    return (Phi3ForCausalLM, (AutoTokenizer, post_load))
