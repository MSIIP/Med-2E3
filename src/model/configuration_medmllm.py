import json
from types import SimpleNamespace

from transformers import AutoConfig, PretrainedConfig


class ConnectorConfig(PretrainedConfig):
    def __init__(
        self, model_arguments, llm_config, encoder_config, encoder_type="vision2d"
    ):
        self.input_dim = encoder_config.hidden_size
        self.output_dim = llm_config.hidden_size

        if encoder_type == "vision2d":
            self.connector_type = model_arguments.vision2d_connector_type
            self.img_size = [encoder_config.image_size] * 2
            self.patch_size = [encoder_config.patch_size] * 2
        elif encoder_type == "vision3d":
            self.connector_type = model_arguments.vision3d_connector_type
            self.img_size = encoder_config.img_size
            self.patch_size = encoder_config.patch_size


class MedMLLMConfig(PretrainedConfig):
    def __init__(self, model_arguments=None, **kwargs):
        # 判断model_arguments是否需要从json文件中加载
        if model_arguments is not None and isinstance(model_arguments, str):
            with open(model_arguments, "r") as f:
                model_arguments = json.load(f)
            model_arguments = self.format_args(model_arguments)

        if model_arguments is not None:
            self.cache_dir_hf = model_arguments.cache_dir_hf
            self.exp_id = model_arguments.exp_id

            self.tokenizer_name_or_path = model_arguments.tokenizer_name_or_path
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = model_arguments.llm_name_or_path
            self.tokenizer_use_fast = model_arguments.tokenizer_use_fast
            self.llm_type = model_arguments.llm_type
            self.llm_max_length = model_arguments.llm_max_length
            self.llm_padding_side = model_arguments.llm_padding_side
            self.llm_attn_implementation = model_arguments.llm_attn_implementation

            self.vision2d_model_type = model_arguments.vision2d_model_type
            self.vision2d_model_select_layer = model_arguments.vision2d_model_select_layer
            self.vision2d_model_select_feature = model_arguments.vision2d_model_select_feature
            self.vision2d_connector_type = model_arguments.vision2d_connector_type

            self.vision3d_model_type = model_arguments.vision3d_model_type
            self.vision3d_model_select_layer = model_arguments.vision3d_model_select_layer
            self.vision3d_model_select_feature = model_arguments.vision3d_model_select_feature
            self.vision3d_connector_type = model_arguments.vision3d_connector_type

            self.llm_config = self.load_llm_config(model_arguments)
            self.vision2d_model_config = self.load_vision2d_model_config(model_arguments)
            self.vision2d_connector_config = ConnectorConfig(
                model_arguments,
                self.llm_config,
                self.vision2d_model_config,
                encoder_type="vision2d"
            )
            self.vision3d_model_config = self.load_vision3d_model_config(model_arguments)
            self.vision3d_connector_config = ConnectorConfig(
                model_arguments,
                self.llm_config,
                self.vision3d_model_config,
                encoder_type="vision3d"
            )

            # only for deepspeed
            self.hidden_size = self.llm_config.hidden_size

        super().__init__(**kwargs)

    def load_llm_config(self, model_arguments):
        llm_config = AutoConfig.from_pretrained(
            model_arguments.llm_name_or_path,
            cache_dir=model_arguments.cache_dir_hf,
            trust_remote_code=True,  # for phi3
        )
        return llm_config

    def load_vision2d_model_config(self, model_arguments):
        vision2d_model_config = AutoConfig.from_pretrained(
            model_arguments.vision2d_model_name_or_path,
            cache_dir=model_arguments.cache_dir_hf,
            trust_remote_code=True,
        )
        if hasattr(vision2d_model_config, "vision_config"):
            _name_or_path = vision2d_model_config._name_or_path
            vision2d_model_config = getattr(vision2d_model_config, "vision_config", vision2d_model_config)
            vision2d_model_config._name_or_path = _name_or_path
        # with open("/mnt/nfs_share/shiym/ckpts/cache_dir_hf/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
        #     def dict_to_namespace(d):
        #         for key, value in d.items():
        #             if isinstance(value, dict):
        #                 d[key] = dict_to_namespace(value)
        #         return SimpleNamespace(**d)

        #     config = json.load(f)
        #     vision2d_model_config = config["model_cfg"]["vision_cfg"]
        #     vision2d_model_config["hidden_size"] = config["model_cfg"]["embed_dim"]
        #     vision2d_model_config["patch_size"] = 16

        # vision2d_model_config.hidden_size = 768
        # vision2d_model_config.image_size = 224
        # vision2d_model_config.patch_size = 16
        return vision2d_model_config

    def load_vision3d_model_config(self, model_arguments):
        vision3d_model_config = AutoConfig.from_pretrained(
            model_arguments.vision3d_model_name_or_path,
            cache_dir=model_arguments.cache_dir_hf,
            trust_remote_code=True,
        )
        return vision3d_model_config

    def format_args(self, model_arguments):
        def dict_to_namespace(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)

        model_arguments = dict_to_namespace(model_arguments)
        model_arguments.llm_name_or_path = model_arguments.llm_config._name_or_path
        model_arguments.vision2d_model_name_or_path = model_arguments.vision2d_model_config._name_or_path
        model_arguments.vision3d_model_name_or_path = model_arguments.vision3d_model_config._name_or_path
        return model_arguments