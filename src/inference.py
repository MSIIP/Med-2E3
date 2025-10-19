import argparse
import json
import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_data_module
from src.model.configuration_medmllm import MedMLLMConfig
from src.model.modeling_medmllm import MedMLLMForConditionalGeneration


def load_model(args):
    model_config = MedMLLMConfig(osp.join(args.resume_from_checkpoint, "args/config.json"))
    model_config.cache_dir_hf = args.cache_dir_hf
    # model_config.llm_attn_implementation = "eager"
    model = MedMLLMForConditionalGeneration(model_config)

    llm_path = osp.join(args.resume_from_checkpoint, "llm")
    vision2d_model_path = osp.join(args.resume_from_checkpoint, "vision2d_model")
    vision2d_connector_path = osp.join(args.resume_from_checkpoint, "vision2d_connector")
    vision3d_model_path = osp.join(args.resume_from_checkpoint, "vision3d_model")
    vision3d_connector_path = osp.join(args.resume_from_checkpoint, "vision3d_connector")

    model.load_vision2d_model(vision2d_model_path)
    model.load_vision2d_connector(vision2d_connector_path)
    model.load_vision3d_model(vision3d_model_path)
    model.load_vision3d_connector(vision3d_connector_path)

    if osp.exists(osp.join(args.resume_from_checkpoint, "adapter_config.json")):
        model.llm = model.llm.from_pretrained(
            pretrained_model_name_or_path=model_config.llm_config._name_or_path,
            torch_dtype=torch.float16 if args.llm_dtype == "float16" else (torch.bfloat16 if args.llm_dtype == "bfloat16" else torch.float32),
            attn_implementation=model_config.llm_attn_implementation,
            cache_dir=model_config.cache_dir_hf,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint)
        model = model.merge_and_unload()
    else:
        model.load_llm(
            llm_path,
            torch.float16 if args.llm_dtype == "float16" else (torch.bfloat16 if args.llm_dtype == "bfloat16" else torch.float32),
            model_config.llm_attn_implementation,
        )

    return model


def inference(args):
    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load model...")
    model = load_model(args)

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Create data_module...")
    data_module = create_data_module(
        model=model,
        data_arguments=args,
        mode="eval"
    )
    data_loader = DataLoader(
        data_module["train_dataset"],
        batch_size=1,
        shuffle=False,
        collate_fn=data_module["data_collator"],
        # num_workers=8,
    )

    print("*" * 30 + "Stage 3" + "*" * 30)
    print("Move model to cuda...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if args.llm_dtype == "float16" else (torch.bfloat16 if args.llm_dtype == "bfloat16" else torch.float32)
    model.to(device=device, dtype=model_dtype)
    model.eval()

    print("*" * 30 + "Stage 4" + "*" * 30)
    print("Inference...")
    outputs_list = []
    # attns_list = []
    scores_list = []
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        for k, v in batch.items():
            if v is not None:
                batch[k] = v.to(device)
                if k == "vision2d" or k == "vision3d" or k == "vision3d_224":
                    batch[k] = v.to(device=device, dtype=model_dtype)

        # from thop import profile
        # flops, params = profile(model, inputs=(batch["input_ids"], batch["attention_mask"], None, batch["vision3d"], None, None, batch["questions"], batch["questions_mask"],))
        # print(f"FLOPs: {flops}, Params: {params}")

        output_ids, scores = model.generate(
            **batch,
            max_length=model.config.llm_max_length if args.max_length is None else args.max_length,
            do_sample=True if args.temperature > 0 else False,
            num_beams=args.num_beams,
            temperature=args.temperature,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )

        outputs = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs_list.append(dict(idx=idx, outputs=outputs))
        # attns_list.append(attns.to("cpu"))
        if scores is not None:
            scores = scores.squeeze().to("cpu")
            scores_list.append(scores)
            print("final scores:", scores.shape, scores)
        print(outputs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, osp.basename(args.data_path)), "w") as f:
        json.dump(outputs_list, f)

    # save attns
    # filename = f"attns_{osp.basename(args.data_path).split('.')[0]}.pt"
    # attns_path = osp.join(args.output_dir, filename)
    # torch.save(attns_list, attns_path)

    # save scores
    filename = f"scores_{osp.basename(args.data_path).split('.')[0]}.pt"
    scores_path = osp.join(args.output_dir, filename)
    torch.save(scores_list, scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir_hf", type=str, default="~/.cache/huggingface")
    parser.add_argument("--llm_dtype", type=str, default="float16")
    parser.add_argument("--data_path", type=str, default="path/to/inference.json")
    parser.add_argument("--conv_version", type=str, default="phi")
    parser.add_argument("--vision2d_data_path", type=str, default="/")
    parser.add_argument("--vision3d_data_path", type=str, default="/")
    parser.add_argument("--resume_from_checkpoint", type=str, default="work_dirs/Med-2E3-finetune")
    parser.add_argument("--output_dir", type=str, default="work_dirs/Med-2E3-finetune/eval")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    inference(args)
