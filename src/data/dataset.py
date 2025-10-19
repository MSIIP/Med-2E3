import copy
import json
import os.path as osp
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import SiglipTextModel, AutoTokenizer

from src.data.preprocess.vision2d_preprocess import Vision2DPreprocess
from src.data.preprocess.vision3d_preprocess import Vision3DPreprocess
from src.data.text_preprocess import TextPreprocess


class MultiModalDataset(Dataset):
    def __init__(self, model, data_arguments, mode="train"):
        super(MultiModalDataset, self).__init__()
        with open(data_arguments.data_path, "r") as f:
            self.data = json.load(f)

        self.data_arguments = data_arguments
        self.text_preprocess = TextPreprocess(
            model.tokenizer, data_arguments.conv_version
        )
        self.vision2d_preprocess = Vision2DPreprocess(
            model.vision2d_model.vision2d_processor, data_arguments
        )
        self.vision3d_preprocess = Vision3DPreprocess(
            model.vision3d_model.vision3d_processor, data_arguments, mode
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            cache_dir="/mnt/nfs_share/shiym/ckpts/cache_dir_hf",
        )

        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_dict = self.text_preprocess(
            messages=copy.deepcopy(data_item["conversations"]),
            mode=self.mode,
        )
        # print("-" * 30 + "item" + "-" * 30)
        # print("before preprocess" + "-" * 29)
        # for k, v in data_item.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v[:10]
        #     print(f"{k:16}", v)
        # print("after preprocess" + "-" * 30)
        # for k, v in data_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v[:10]
        #     print(f"{k:16}", v)

        if "vision2d" in data_item:  # for multi vision2d
            data_dict["vision2d"] = []
            for filename in data_item["vision2d"]:
                vision2d_path = osp.join(self.data_arguments.vision2d_data_path, filename)
                vision2d = Image.open(vision2d_path).convert("RGB")
                vision2d = self.vision2d_preprocess(vision2d)
                data_dict["vision2d"].append(vision2d)
        else:
            data_dict["vision2d"] = None

        if "vision3d" in data_item:
            data_dict["vision3d"] = []
            data_dict["vision3d_224"] = None
            for filename in data_item["vision3d"]:
                vision3d_path = osp.join(self.data_arguments.vision3d_data_path, filename)
                vision3d = np.load(vision3d_path)
                # vision3d_224_path = osp.join(self.data_arguments.vision3d_data_path + "_224", filename)
                # vision3d_224 = np.load(vision3d_224_path)
                # vision3d_path2 = osp.join(self.data_arguments.vision3d_data_path + "_2", filename)
                # vision3d2 = np.load(vision3d_path2)
                # vision3d = np.concatenate([vision3d, vision3d1, vision3d2], axis=1)  # (1, 96, 256, 256)
                vision3d = self.vision3d_preprocess(vision3d)
                data_dict["vision3d"].append(vision3d)
                # vision3d_224 = self.vision3d_preprocess(vision3d_224)
                # data_dict["vision3d_224"].append(vision3d_224)
        else:
            data_dict["vision3d"] = None
            data_dict["vision3d_224"] = None

        # for question_type: 0-10
        if "question_type" in data_item:
            data_dict["question_type"] = data_item["question_type"]

        if "question" in data_item:
            # if self.text_preprocess.tokenizer("test string").input_ids[0] == self.text_preprocess.tokenizer.bos_token_id:
            #     offset = 1
            # else:
            #     offset = 0

            # question = data_item["question"]
            # question = self.text_preprocess.tokenizer(question).input_ids[offset:]
            # data_dict["questions"] = torch.tensor(question, dtype=torch.long)
            text_tensor = self.tokenizer(data_item["question"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
            data_dict["questions"] = text_tensor["input_ids"][0]
            data_dict["questions_mask"] = text_tensor["attention_mask"][0]

        return data_dict


class DataCollatorForMultiModalDataset:
    def __init__(self, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # questions = [instance["questions"] for instance in instances]
        # questions = nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # questions = questions[:, : self.tokenizer.model_max_length]
        # questions_mask = questions.ne(self.tokenizer.pad_token_id)
        # batch["questions"] = questions
        # batch["questions_mask"] = questions_mask
        questions = [instance["questions"] for instance in instances]
        questions_mask = [instance["questions_mask"] for instance in instances]
        batch["questions"] = torch.stack(questions)
        batch["questions_mask"] = torch.stack(questions_mask)

        if self.mode == "train":
            labels = [instance["labels"] for instance in instances]
            if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                for label in labels:
                    label[label == self.tokenizer.eos_token_id] = -300

            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = labels[:, : self.tokenizer.model_max_length]

            if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                for label in labels:
                    label[label == -300] = self.tokenizer.eos_token_id
            batch["labels"] = labels

        vision2d_list = []
        for instance in instances:
            if instance["vision2d"] is not None:
                vision2d_list.extend(instance["vision2d"])  # for multi vision2d
        vision2d = torch.stack(vision2d_list) if len(vision2d_list) > 0 else None
        batch["vision2d"] = vision2d

        vision3d_list = []
        for instance in instances:
            if instance["vision3d"] is not None:
                vision3d_list.extend(instance["vision3d"])
        vision3d = torch.stack(vision3d_list) if len(vision3d_list) > 0 else None
        batch["vision3d"] = vision3d

        vision3d_224_list = []
        for instance in instances:
            if instance["vision3d_224"] is not None:
                vision3d_224_list.extend(instance["vision3d_224"])
        vision3d_224 = torch.stack(vision3d_224_list) if len(vision3d_224_list) > 0 else None
        batch["vision3d_224"] = vision3d_224

        # for question_type: 0-10
        if "question_type" in instances[0]:
            batch["question_type"] = torch.tensor([instance["question_type"] for instance in instances])

        return batch


def create_data_module(model, data_arguments, mode="train"):
    train_dataset = MultiModalDataset(model=model, data_arguments=data_arguments, mode=mode)
    data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode=mode)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
