import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.model.encoder.connector import CONNECTOR_FACTORY
from src.model.encoder.vision2d import VISION2D_FACTORY
from src.model.encoder.vision3d import VISION3D_FACTORY
from src.model.llm import LLM_FACTORY
from src.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_ID, IMAGE3D_TOKEN_ID


class MedMLLMPreTrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.llm_config.initializer_range

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MedMLLMForConditionalGeneration(MedMLLMPreTrainedModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        llm, (tokenizer, post_load) = LLM_FACTORY[model_config.llm_type]
        self.tokenizer = post_load(
            tokenizer.from_pretrained(
                model_config.tokenizer_name_or_path,
                model_max_length=model_config.llm_max_length,
                padding_side=model_config.llm_padding_side,
                cache_dir=model_config.cache_dir_hf,
                use_fast=model_config.tokenizer_use_fast,
                trust_remote_code=True,
            )
        )

        self.llm = llm(model_config.llm_config)
        self.vision2d_model = VISION2D_FACTORY[model_config.vision2d_model_type](
            vision2d_model_config=model_config.vision2d_model_config,
            cache_dir_hf=model_config.cache_dir_hf,
        )
        self.vision2d_connector = CONNECTOR_FACTORY[model_config.vision2d_connector_type](model_config.vision2d_connector_config)
        self.vision3d_model = VISION3D_FACTORY[model_config.vision3d_model_type](
            vision3d_model_config=model_config.vision3d_model_config,
            cache_dir_hf=model_config.cache_dir_hf,
        )
        self.vision3d_connector = CONNECTOR_FACTORY[model_config.vision3d_connector_type](model_config.vision3d_connector_config)

        self.post_init()
        self.current_lr = 0
        self.lr_ratio = 0

    @torch.no_grad()
    def generate(
        self, input_ids, attention_mask, vision2d=None, vision3d=None, question_type=None,
        questions=None, questions_mask=None, vision3d_224=None, questions_pooled=None, **generation_config,
    ):
        llm_inputs = self.prepare_inputs_for_multimodal(
            input_ids, attention_mask, vision2d, vision3d, question_type=question_type,
            questions=questions, questions_mask=questions_mask, vision3d_224=vision3d_224, questions_pooled=questions_pooled,
        )
        if llm_inputs["inputs_embeds"] is None:
            llm_inputs["inputs_embeds"] = self.llm.get_input_embeddings()(llm_inputs["input_ids"])
        llm_inputs["inputs_embeds"] = llm_inputs["inputs_embeds"].to(dtype=self.llm.dtype)
        del llm_inputs["input_ids"]
        del llm_inputs["pos_vision"]
        scores = llm_inputs["scores"]
        del llm_inputs["scores"]

        llm_inputs["use_cache"] = True    
        llm_outputs = self.llm.generate(**llm_inputs, **generation_config)
        # # attentions: len_ans * num_layer * (B, num_head, len_prompt/1, len_prompt)
        # pos_vision = llm_inputs["pos_vision"][0]
        # del llm_inputs["pos_vision"]
        # llm_outputs = self.llm.generate(**llm_inputs, **generation_config, output_attentions=True, return_dict_in_generate=True)
        # # extract attention scores
        # attentions = llm_outputs.attentions
        # attns = []
        # for attn_per_token in attentions:
        #     attn_per_token_new = []
        #     for attn_per_layer in attn_per_token:
        #         attn_per_token_new.append(attn_per_layer[0, :, 0, pos_vision: pos_vision+512])
        #     attn_per_token_new = torch.stack(attn_per_token_new, dim=0)
        #     attns.append(attn_per_token_new)
        # attns = torch.stack(attns, dim=0)
        # print(
        #     len(attentions),
        #     len(attentions[-1]),
        #     attentions[-1][0].shape,
        #     llm_inputs["inputs_embeds"].shape,
        #     pos_vision,
        #     attns.shape,  # (len_ans, num_layer, num_head, 512)
        # )
        # llm_outputs = llm_outputs.sequences

        return llm_outputs, scores

    def forward(
        self, input_ids, attention_mask, vision2d=None, vision3d=None, labels=None, question_type=None,
        questions=None, questions_mask=None, vision3d_224=None, questions_pooled=None, **kwargs,
    ):
        llm_inputs = self.prepare_inputs_for_multimodal(
            input_ids, attention_mask, vision2d, vision3d, labels, question_type=question_type,
            questions=questions, questions_mask=questions_mask, vision3d_224=vision3d_224, questions_pooled=questions_pooled,
        )
        if llm_inputs["inputs_embeds"] is not None:
            llm_inputs["inputs_embeds"] = llm_inputs["inputs_embeds"].to(dtype=self.llm.dtype)
        del llm_inputs["pos_vision"]
        del llm_inputs["scores"]

        if hasattr(self.config, "use_cache"):
            llm_inputs["use_cache"] = False
        llm_outputs = self.llm(**llm_inputs)
        return llm_outputs

    def prepare_inputs_for_multimodal(
        self, input_ids, attention_mask, vision2d, vision3d, labels=None, question_type=None,
        questions=None, questions_mask=None, vision3d_224=None, questions_pooled=None,
    ):
        if vision2d is None and vision3d is None and vision3d_224 is None:
            return dict(
                input_ids=input_ids,
                inputs_embeds = None,
                attention_mask=attention_mask,
                labels=labels,
            )

        # encode vision2d
        if vision2d is not None:
            vision2d = self.vision2d_model(
                vision2d,
                vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                vision2d_model_select_feature=self.config.vision2d_model_select_feature,
            )
            vision2d = self.vision2d_connector(vision2d)

        scores = None

        # encode vision3d
        if vision3d is not None:
            if self.config.exp_id == 0:
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )
                vision3d = self.vision3d_connector(vision3d)
            elif self.config.exp_id == 1:
                # (B, C, N, H, W) for 单轴位增强+平均分
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                # vision2d, pooler_output = self.vision2d_model(
                #     vision2d,
                #     vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                #     vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                # )
                # pooler_output = pooler_output.view(B, N, pooler_output.shape[1])  # (B, N, D)
                # scores = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
                # scores = torch.bmm(scores, questions_pooled.unsqueeze(-1))  # (B, N, 1)
                # scores = F.softmax(scores, dim=1)
                # 归一化
                # scores = scores / scores.sum(dim=1, keepdim=True)  # (B, N, 1)

                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)
                # pooler_output = self.vision2d_connector(pooler_output)  # (B*N, D)
                # pooler_output = pooler_output.view(B, N, pooler_output.shape[1])  # (B, N, D)

                # process vision3d
                vision3d = self.vision3d_connector(self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                ))  # (B, L, D)

                # all
                vision2d = vision2d.reshape(B, N*vision2d.shape[2], vision2d.shape[3])  # (B, N*L, D)
                # pooling
                # AvgPooling
                # vision2d = vision2d.mean(dim=1)  # (B, L, D)
                # MaxPooling
                # vision2d = vision2d.max(dim=1)[0]  # (B, L, D)

                # scoring
                # # randn-32
                # scores = torch.randn(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores = scores
                # scores = F.softmax(scores, dim=1)
                # vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                # vision2d = vision2d.sum(dim=1)  # (B, L, D)
                # # gauss scores
                # indices = torch.arange(N).float()  # 从0到N-1的索引
                # mid = (N - 1) / 2  # 找到中间位置
                # scores = torch.exp(-(indices - mid) ** 2 / (2 * (N / 6) ** 2))  # 高斯分布公式
                # scores = scores / scores.sum()
                # scores = scores.unsqueeze(0).repeat(B, 1).unsqueeze(-1).to(device=self.device, dtype=vision3d.dtype)  # (B, N, 1)
                # vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                # vision2d = vision2d.sum(dim=1)  # (B, L, D)
                # # TG-IS
                # questions_embed = self.llm.get_input_embeddings()(questions)
                # questions_embed = questions_embed * questions_mask.unsqueeze(-1)
                # questions_embed = questions_embed.sum(dim=1) / questions_mask.sum(dim=1).unsqueeze(-1)
                # B, L, D = vision3d.shape
                # scores = vision3d.reshape(B, 4, L//4, D)
                # scores = scores.repeat_interleave(8, dim=1)
                # scores = torch.cat([scores, vision2d], dim=2)
                # scores = scores.mean(dim=2)
                # scores = torch.bmm(scores, questions_embed.unsqueeze(-1))
                # scores = F.softmax(scores, dim=1)
                # vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                # vision2d = vision2d.sum(dim=1)  # (B, L, D)

                print(f"vision2d: {vision2d.shape}, vision3d: {vision3d.shape}")
                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)
                # vision3d = vision2d

            # for 单轴位增强+学习分数
            elif self.config.exp_id == 2:
                # (B, C, N, H, W)
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)
                # 通过平均变成(B, N//4, L, D)
                vision2d = vision2d.view(B, N//4, 4, vision2d.shape[2], vision2d.shape[3]).mean(dim=2)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                # temperature = self.lr_ratio * 50
                # if temperature < 10:
                #     temperature = 10
                # print(f"lr_ratio: {self.lr_ratio}, temperature: {temperature}, current_lr: {self.current_lr}")
                vision3d, scores = self.vision3d_connector(vision3d, question_type, temperature=1)

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N//4, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)
                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)

            elif self.config.exp_id == 3:
                # (B, C, N, H, W)
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                vision3d, scores = self.vision3d_connector(vision3d, question_type, temperature=1, vision2d=vision2d)

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)
                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)

            elif self.config.exp_id == 4:
                # (B, C, N, H, W)
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                # TG-IS
                # questions_embed = self.llm.get_input_embeddings()(questions)
                # questions_embed = questions_embed * questions_mask.unsqueeze(-1)
                # questions_embed = questions_embed.sum(dim=1) / questions_mask.sum(dim=1).unsqueeze(-1)
                questions_embed = self.vision3d_model.vision3d_encoder.language_encoder(questions, questions_mask).pooler_output
                vision3d, scores = self.vision3d_connector(vision3d, question_type, temperature=1, vision2d=vision2d, text=questions_embed)

                # # random select 3 slices
                # scores = torch.rand(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores = scores
                # scores = F.softmax(scores, dim=1)
                # scores = torch.zeros(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # indices = torch.randperm(32)[:3]  # 随机选择3个索引
                # scores[0, indices] = 1/3
                # top_k_values, top_k_indices = torch.topk(scores, k=3, dim=1)
                # scores = torch.zeros(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores.scatter_(1, top_k_indices, 1 / 3)

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)

                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)
                # vision3d = vision2d

            elif self.config.exp_id == 5:
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                # process text
                questions_embed = self.llm.get_input_embeddings()(questions)
                vision3d, scores = self.vision3d_connector(
                    vision3d, question_type, temperature=1, vision2d=vision2d, text=questions_embed, mask=questions_mask,
                )

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)
                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)

            elif self.config.exp_id == 6:
                # for biomedclip
                # process vision2d
                B, C, N, H, W = vision3d_224.size()
                vision2d = vision3d_224.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                # process text
                questions_embed = self.llm.get_input_embeddings()(questions)
                questions_embed = questions_embed * questions_mask.unsqueeze(-1)
                questions_embed = questions_embed.sum(dim=1) / questions_mask.sum(dim=1).unsqueeze(-1)
                vision3d, scores = self.vision3d_connector(vision3d, question_type, temperature=1, vision2d=vision2d, text=questions_embed)

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)

                # 2d3d avg
                # vision2d = vision2d.mean(dim=1)  # (B, L, D)

                # 2d3d max
                # vision2d = vision2d.max(dim=1)[0]  # (B, L, D)

                # 2d3d rand
                # scores = torch.randn(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores = F.softmax(scores, dim=1)
                # vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                # vision2d = vision2d.sum(dim=1)  # (B, L, D)

                vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)
                # vision3d = vision2d
        
            elif self.config.exp_id == 7:
                # (B, C, N, H, W)
                B, C, N, H, W = vision3d.size()

                # process vision2d
                vision2d = vision3d.transpose(1, 2).contiguous()  # (B, N, C, H, W)
                vision2d = vision2d.view(B*N, C, H, W)  # (B*N, C, H, W)
                vision2d = vision2d.expand(B*N, 3, H, W)  # (B*N, 3, H, W)
                vision2d, pooler_output = self.vision2d_model(
                    vision2d,
                    vision2d_model_select_layer=self.config.vision2d_model_select_layer,
                    vision2d_model_select_feature=self.config.vision2d_model_select_feature,
                )
                vision2d = self.vision2d_connector(vision2d)  # (B*N, L, D)
                vision2d = vision2d.view(B, N, vision2d.shape[1], vision2d.shape[2])  # (B, N, L, D)
                pooler_output = self.vision2d_connector(pooler_output)  # (B*N, D)
                pooler_output = pooler_output.view(B, N, pooler_output.shape[1])  # (B, N, D)

                # process vision3d
                vision3d = self.vision3d_model(
                    vision3d,
                    vision3d_model_select_layer=self.config.vision3d_model_select_layer,
                    vision3d_model_select_feature=self.config.vision3d_model_select_feature,
                )

                # process text
                questions_embed = self.llm.get_input_embeddings()(questions)
                questions_embed = questions_embed * questions_mask.unsqueeze(-1)
                questions_embed = questions_embed.sum(dim=1) / questions_mask.sum(dim=1).unsqueeze(-1)
                vision3d, scores = self.vision3d_connector(vision3d, question_type, temperature=1, vision2d=vision2d, text=questions_embed)

                # new code
                # scores = torch.rand(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores = scores
                # scores = F.softmax(scores, dim=1)
                # scores = torch.zeros(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # indices = torch.randperm(32)[:3]  # 随机选择3个索引
                # scores[0, indices] = 1/3
                # top_k_values, top_k_indices = torch.topk(scores, k=3, dim=1)
                # scores = torch.zeros(B, 32, 1, device=self.device, dtype=vision3d.dtype)
                # scores.scatter_(1, top_k_indices, 1 / 3)

                # 加权平均
                vision2d = vision2d * scores.unsqueeze(-1)  # (B, N, L, D)
                vision2d = vision2d.sum(dim=1)  # (B, L, D)
                vision3d = vision2d
                # vision3d = torch.cat([vision3d, vision2d], dim=1)  # (B, L+L, D)

        # 初始化labels
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX, device=self.device)

        # 根据attention_mask筛选input_ids、labels，转换为list格式
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # 将vision2d和vision3d插入到input_ids、labels中，得到inputs_embeds、new_labels，支持多图操作
        cur_vision2d_idx = 0
        cur_vision3d_idx = 0
        inputs_embeds = []
        new_labels = []
        pos_vision = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_vision2d = (cur_input_ids == IMAGE_TOKEN_ID).sum()
            num_vision3d = (cur_input_ids == IMAGE3D_TOKEN_ID).sum()
            if num_vision2d == 0 and num_vision3d == 0:
                inputs_embeds.append(self.llm.get_input_embeddings()(cur_input_ids))
                new_labels.append(labels[batch_idx])
                continue

            cur_labels = labels[batch_idx]
            vision2d_token_idx_list = torch.where(cur_input_ids == IMAGE_TOKEN_ID)[0].tolist()
            vision3d_token_idx_list = torch.where(cur_input_ids == IMAGE3D_TOKEN_ID)[0].tolist()
            special_token_idx_list = sorted(vision2d_token_idx_list + vision3d_token_idx_list+ [-1] + [cur_input_ids.size(0)])
            cur_input_ids_text = []
            cur_labels_text = []

            # 将cur_input_ids、cur_labels按照vision2d和vision3d的位置分割，得到cur_input_ids_text、cur_labels_text
            special_token_type = []
            for i in range(len(special_token_idx_list) - 1):
                start_idx = special_token_idx_list[i] + 1
                end_idx = special_token_idx_list[i + 1]
                cur_input_ids_text.append(cur_input_ids[start_idx:end_idx])
                cur_labels_text.append(cur_labels[start_idx:end_idx])

                if end_idx in vision2d_token_idx_list:
                    special_token_type.append("vision2d")
                elif end_idx in vision3d_token_idx_list:
                    special_token_type.append("vision3d")

            # 将cur_input_ids_text转换为cur_inputs_embeds_text，并按照vision2d和vision3d的位置切分
            split_lengths = [x.size(0) for x in cur_input_ids_text]
            cur_inputs_embeds_text = self.llm.get_input_embeddings()(torch.cat(cur_input_ids_text))
            cur_inputs_embeds_text = torch.split(cur_inputs_embeds_text, split_lengths)

            # 按照位置在cur_inputs_embeds_text中插入vision2d和vision3d，在cur_labels_text中插入IGNORE_INDEX，得到cur_inputs_embeds、cur_new_labels
            cur_inputs_embeds = []
            cur_new_labels = []
            for i in range(num_vision2d + num_vision3d + 1):
                cur_inputs_embeds.append(cur_inputs_embeds_text[i])
                cur_new_labels.append(cur_labels_text[i])

                if i < num_vision2d + num_vision3d:
                    pos_vision.append(cur_inputs_embeds_text[i].shape[-2])
                    if special_token_type[i] == "vision2d":
                        cur_inputs_embeds.append(vision2d[cur_vision2d_idx])
                        cur_new_labels.append(
                            torch.full(
                                (vision2d[cur_vision2d_idx].size(0),),
                                IGNORE_INDEX,
                                device=self.device,
                            )
                        )
                        cur_vision2d_idx += 1
                    elif special_token_type[i] == "vision3d":
                        cur_inputs_embeds.append(vision3d[cur_vision3d_idx])
                        cur_new_labels.append(
                            torch.full(
                                (vision3d[cur_vision3d_idx].size(0),),
                                IGNORE_INDEX,
                                device=self.device,
                            )
                        )
                        cur_vision3d_idx += 1

            # 将cur_inputs_embeds、cur_new_labels合并到inputs_embeds、new_labels
            inputs_embeds.append(torch.cat(cur_inputs_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        # 将inputs_embeds、new_labels按照llm_max_length截断
        inputs_embeds = [x[: self.config.llm_max_length] for x in inputs_embeds]
        labels = [x[: self.config.llm_max_length] for x in new_labels]

        # 初始化inputs_embeds_padded、labels_padded、attention_mask_padded、position_ids_padded
        batch_size = len(inputs_embeds)
        max_len = max(x.size(0) for x in inputs_embeds)
        inputs_embeds_padded = []
        labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            device=self.device,
        )
        attention_mask_padded = torch.zeros(
            (batch_size, max_len),
            device=self.device,
        )

        # 按照llm_padding_side填充inputs_embeds_padded、labels_padded、attention_mask_padded、position_ids_padded
        for i, (cur_inputs_embeds, cur_labels) in enumerate(zip(inputs_embeds, labels)):
            cur_len = cur_inputs_embeds.size(0)
            zero_inputs_embeds = torch.zeros(
                (max_len - cur_len, cur_inputs_embeds.size(1)),
                device=self.device,
            )
            if self.config.llm_padding_side == "left":
                inputs_embeds_padded.append(torch.cat((zero_inputs_embeds, cur_inputs_embeds)))
                labels_padded[i, -cur_len:] = cur_labels
                attention_mask_padded[i, -cur_len:] = True
            else:
                inputs_embeds_padded.append(torch.cat((cur_inputs_embeds, zero_inputs_embeds)))
                labels_padded[i, :cur_len] = cur_labels
                attention_mask_padded[i, :cur_len] = True
        inputs_embeds_padded = torch.stack(inputs_embeds_padded)

        return dict(
            input_ids=None,
            inputs_embeds=inputs_embeds_padded,
            attention_mask=attention_mask_padded,
            labels=labels_padded,
            pos_vision=pos_vision,
            scores=scores,
        )

    def load_llm(self, llm_path, llm_dtype, llm_attn_implementation):
        llm_name_or_path = self.config.llm_config._name_or_path if llm_path is None else llm_path
        self.llm = self.llm.from_pretrained(
            pretrained_model_name_or_path=llm_name_or_path,
            torch_dtype=llm_dtype,
            attn_implementation=llm_attn_implementation,
            cache_dir=self.config.cache_dir_hf,
        )

        self.llm.requires_grad_(False)

    def load_vision2d_model(self, vision2d_model_path):
        vision2d_model_name_or_path = self.config.vision2d_model_config._name_or_path if vision2d_model_path is None else vision2d_model_path
        self.vision2d_model.load_model(vision2d_model_name_or_path)

    def load_vision2d_connector(self, vision2d_connector_path):
        self.vision2d_connector.load_model(vision2d_connector_path)

    def load_vision3d_model(self, vision3d_model_path):
        vision3d_model_name_or_path = self.config.vision3d_model_config._name_or_path if vision3d_model_path is None else vision3d_model_path
        self.vision3d_model.load_model(vision3d_model_name_or_path)

    def load_vision3d_connector(self, vision3d_connector_path):
        self.vision3d_connector.load_model(vision3d_connector_path)