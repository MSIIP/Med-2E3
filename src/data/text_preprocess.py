from src.data.template import TEMPlATE_FACTORY


class TextPreprocess:
    def __init__(self, tokenizer, conv_version):
        self.tokenizer = tokenizer
        self.template = TEMPlATE_FACTORY[conv_version]()

    def __call__(self, messages, mode="train"):
        return self.template.encode(messages, self.tokenizer, mode)
