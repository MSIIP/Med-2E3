import copy
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from src.model.encoder.connector.base import Connector


ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


class PlanePoolingConnector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^plane_pooling_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        seq2cube = Rearrange(
            "b (p1 p2) d -> b d p1 p2",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
        )
        x = seq2cube(x)
        x = F.avg_pool2d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
        
        cube2seq = Rearrange(
            "b d p1 p2 -> b (p1 p2) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
        )
        x = cube2seq(x)
        x = self._connector(x)

        return x


class SpatialPoolingConnector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
        
        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)

        return x


class SpatialPoolingScoreConnector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self._task_embedding = nn.Embedding(11, 8)
        self._image_compression = nn.AdaptiveAvgPool2d(1)

        hidden_dim = connector_config.input_dim
        self._scoring_module = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            ACT_TYPE[act_type](),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, question_type, temperature=1.0):
        q_bak = copy.deepcopy(question_type)
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )

        # scoring
        scores = seq2cube(x)  # (B, D, 8, 16, 16)
        scores = scores.transpose(1, 2).contiguous()  # (B, 8, D, 16, 16)
        scores = scores.view(batch_size * 8, embedding_dim, 16, 16)  # (B*8, D, 16, 16)

        scores = self._image_compression(scores)  # (B*8, D, 1, 1)
        scores = scores.view(batch_size, 8, -1)  # (B, 8, D)
        question_type = self._task_embedding(question_type).unsqueeze(-1)  # (B, 8, 1)
        scores = torch.cat([question_type, scores], dim=2)  # (B, 8, D+1)

        scores = self._scoring_module(scores)  # (B, 8, 1)
        scores = scores / temperature
        scores = F.softmax(scores, dim=1)
        print("question_type:", q_bak[0].squeeze())
        print("question_embedding:", question_type[0].squeeze())
        print("scores:", scores[0].squeeze())

        # pooling
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)

        return x, scores


class SpatialPoolingScore1Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score1_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self.num_task = 11
        self.num_slice = 8
        self.hidden_dim = connector_config.input_dim

        self._task_embedding = nn.Embedding(self.num_task, self.hidden_dim)
        self._image_compression = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, question_type, temperature=1.0):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )

        # scoring
        scores = seq2cube(x).transpose(1, 2).contiguous()  # (B, 8, D, 16, 16)
        scores = scores.view(batch_size * self.num_slice, embedding_dim, 16, 16)  # (B*8, D, 16, 16)
        scores = self._image_compression(scores)  # (B*8, D, 1, 1)
        scores = scores.view(batch_size, self.num_slice, -1)  # (B, 8, D)

        question_type = self._task_embedding(question_type)  # (B, D)
        scores = torch.bmm(scores, question_type.unsqueeze(-1))  # (B, 8, 1)
        scores = scores / temperature
        scores = F.softmax(scores, dim=1)
        print("question_embedding:", question_type.shape, question_type[0].squeeze()[:10])
        print("scores:", scores.shape, scores[0].squeeze())

        # pooling
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)

        return x, scores


class SpatialPoolingScore2Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score2_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self.num_task = 11
        self.num_slice = 8
        self.hidden_dim = connector_config.input_dim

        self._task_embedding = nn.Embedding(self.num_task, self.hidden_dim)
        self._image_compression = nn.AdaptiveAvgPool2d(1)
        self._temperature = nn.Sequential(
            nn.Embedding(self.num_task, 1),
            # nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, question_type, temperature=1.0):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )

        # scoring
        scores = seq2cube(x).transpose(1, 2).contiguous()  # (B, 8, D, 16, 16)
        scores = scores.view(batch_size * self.num_slice, embedding_dim, 16, 16)  # (B*8, D, 16, 16)
        scores = self._image_compression(scores)  # (B*8, D, 1, 1)
        scores = scores.view(batch_size, self.num_slice, -1)  # (B, 8, D)

        temperature = (10 ** (self._temperature(question_type) * 2 - 1)).unsqueeze(-1)  # (B, 1, 1)
        question_type = self._task_embedding(question_type)  # (B, D)
        scores = torch.bmm(scores, question_type.unsqueeze(-1))  # (B, 8, 1)
        scores = scores / temperature  # (B, 8, 1)
        scores = F.softmax(scores, dim=1)
        print("question_embedding:", question_type.shape, question_type[0].squeeze()[:10])
        print("temperature:", temperature.shape, temperature[0].squeeze())
        print("scores:", scores.shape, scores[0].squeeze())

        # pooling
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)

        return x, scores


class SpatialPoolingScore3Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score3_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self.num_task = 11
        self.num_slice = 32
        self.hidden_dim = connector_config.output_dim

        self._task_embedding = nn.Embedding(self.num_task, self.hidden_dim)
        self._temperature = nn.Sequential(
            nn.Embedding(self.num_task, 1),
            # nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, question_type, temperature=1.0, vision2d=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)

        # scoring
        scores = x.unsqueeze(1).expand(batch_size, self.num_slice, x.shape[1], x.shape[2])  # (B, 32, L, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        temperature = (10 ** (self._temperature(question_type) * 2 - 1)).unsqueeze(-1)  # (B, 1, 1)
        question_type = self._task_embedding(question_type)  # (B, D)
        scores = torch.bmm(scores, question_type.unsqueeze(-1))  # (B, 32, 1)
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        print("question_embedding:", question_type.shape, question_type[0].squeeze()[:10])
        print("temperature:", temperature.shape, temperature[0].squeeze())
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore4Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score4_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self.num_task = 11
        self.num_slice = 32
        self.hidden_dim = connector_config.output_dim

        self._task_embedding = nn.Embedding(self.num_task, self.hidden_dim)
        # self._temperature = nn.Sequential(
        #     nn.Embedding(self.num_task, 1),
        #     # nn.Linear(1, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x, question_type, temperature=1.0, vision2d=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)

        # scoring
        scores = x.unsqueeze(1).expand(batch_size, self.num_slice, x.shape[1], x.shape[2])  # (B, 32, L, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        # temperature = (10 ** (self._temperature(question_type) * 2 - 1)).unsqueeze(-1)  # (B, 1, 1)
        question_type = self._task_embedding(question_type)  # (B, D)
        scores = torch.bmm(scores, question_type.unsqueeze(-1))  # (B, 32, 1)
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        print("question_embedding:", question_type.shape, question_type[0].squeeze()[:10])
        # print("temperature:", temperature.shape, temperature[0].squeeze())
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore5Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score5_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)

        # score for each slice
        self.num_task = 11
        self.num_slice = 32
        self.hidden_dim = connector_config.output_dim

        self._task_embedding = nn.Embedding(self.num_task, self.hidden_dim * 2)
        self._slice_embedding = nn.Embedding(self.num_slice, self.hidden_dim)
        # self._temperature = nn.Sequential(
        #     nn.Embedding(self.num_task, 1),
        #     # nn.Linear(1, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x, question_type, temperature=1.0, vision2d=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        # (B, L, D): (16, 8*16*16, 768)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)

        # scoring
        scores = x.unsqueeze(1).expand(batch_size, self.num_slice, x.shape[1], x.shape[2])  # (B, 32, L, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        slice_idx = torch.arange(self.num_slice, device=x.device)  # (32,)
        slice_embedding = self._slice_embedding(slice_idx)  # (32, D)
        slice_embedding = slice_embedding.unsqueeze(0).expand(batch_size, self.num_slice, slice_embedding.shape[1])  # (B, 32, D)
        scores = torch.cat([scores, slice_embedding], dim=-1)  # (B, 32, D*2)

        # temperature = (10 ** (self._temperature(question_type) * 2 - 1)).unsqueeze(-1)  # (B, 1, 1)
        question_type = self._task_embedding(question_type)  # (B, D*2)
        scores = torch.bmm(scores, question_type.unsqueeze(-1))  # (B, 32, 1)
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        print("question_embedding:", question_type.shape, question_type[0].squeeze()[:10])
        # print("temperature:", temperature.shape, temperature[0].squeeze())
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore6Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score6_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        # temperature = 0.05
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore7Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score7_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attention = nn.Linear(connector_config.output_dim, 1)
        self.attn_dropout = nn.Dropout(0.1)
        self.dropout_text = nn.Dropout(0.1)
        self.dropout_vision = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None, mask=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)

        # text (B, L, D)
        attn_weights = self.attention(text)  # (B, L, 1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)  # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, L, 1)
        # attn_weights = self.dropout_text(attn_weights)
        text = torch.sum(attn_weights * text, dim=1)  # (B, D)

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        # scores = scores.mean(dim=2)  # (B, 32, D)
        attn_weights = self.attention(scores)  # (B, 32, L//4+L, 1)
        attn_weights = F.softmax(attn_weights, dim=2)  # (B, 32, L//4+L, 1)
        # attn_weights = self.dropout_vision(attn_weights)
        scores = torch.sum(attn_weights * scores, dim=2)  # (B, 32, D)

        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        # scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore8Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score8_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)
        self.text_query = nn.Parameter(torch.randn(1, 1, connector_config.output_dim))
        self.text_ln_q = nn.LayerNorm(connector_config.output_dim)
        self.text_attn = nn.MultiheadAttention(
            embed_dim=connector_config.output_dim,
            num_heads=8,
            batch_first=True,
        )
        self.vision_query = nn.Parameter(torch.randn(1, 1, connector_config.output_dim))
        self.vision_ln_q = nn.LayerNorm(connector_config.output_dim)
        self.vision_attn = nn.MultiheadAttention(
            embed_dim=connector_config.output_dim,
            num_heads=8,
            batch_first=True,
        )

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None, mask=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)

        B, L, D = x.shape

        # text (B, L, D)
        attn_mask = torch.where(mask, torch.tensor(0.0, device=mask.device), torch.tensor(-1e9, device=mask.device)).unsqueeze(1)  # (B, 1, L)
        attn_mask = attn_mask.repeat(8, 1, 1).to(dtype=x.dtype)  # (B*num_heads, 1, L)
        text, _ = self.text_attn(
            self.text_ln_q(self.text_query).expand(B, -1, -1),
            text,
            text,
            attn_mask=attn_mask,
        )  # (B, 1, D)
        text = text.squeeze(1)  # (B, D)
        print("text:", text.shape, text[0].squeeze()[:8])

        # scoring
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.reshape(B*32, L//4+L, D)
        scores, _ = self.vision_attn(
            self.vision_ln_q(self.vision_query).expand(B*32, -1, -1),
            scores,
            scores,
        )  # (B*32, 1, D)
        scores = scores.view(B, 32, D)  # (B, 32, D)
        print("vision:", scores.shape, scores[0].squeeze()[:3, :8])

        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore9Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score9_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.scoring_module = nn.Linear(connector_config.output_dim, 1)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        # scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        scores = self.scoring_module(scores)  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        # temperature = 0.05
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore10Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score10_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)

        self.text_dim = 768
        self.text_mlp = nn.Linear(self.text_dim, connector_config.output_dim)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        # print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        # scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        text = self.text_mlp(text)
        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        temperature = 5
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore11Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score11_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)

        self.text_dim = 768
        self.text_mlp = nn.Linear(self.text_dim, connector_config.output_dim)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        # print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        # scores = x.reshape(B, 4, L//4, D)
        # scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        # scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = vision2d  # (B, 32, L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        text = self.text_mlp(text)
        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        temperature = 5
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore12Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score12_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        print("question_type:", question_type.shape, question_type[0].squeeze())
        print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.unsqueeze(1).expand(B, 32, L, D)  # (B, 32, L, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        # temperature = 0.05
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore13Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score13_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.scoring_module = nn.Linear(connector_config.output_dim, 1)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        # scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        # scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        scores = self.scoring_module(scores)  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        # temperature = 0.05
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore14Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score14_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.scoring_module = nn.Linear(connector_config.output_dim, 1)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        # scores = x.reshape(B, 4, L//4, D)
        # scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        # scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = vision2d
        scores = scores.mean(dim=2)  # (B, 32, D)

        # scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        scores = self.scoring_module(scores)  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        # temperature = 0.05
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores


class SpatialPoolingScore15Connector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        img_size = connector_config.img_size
        patch_size = connector_config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(img_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^spatial_pooling_score15_mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
        self.attn_dropout = nn.Dropout(0.1)

        self.text_dim = 768
        self.text_mlp = nn.Linear(self.text_dim, connector_config.output_dim)

    def forward(self, x, question_type, temperature=1.0, vision2d=None, text=None):
        # print("question_type:", question_type.shape, question_type[0].squeeze())
        # print("text:", text.shape, text[0].squeeze()[:10])  # (B, D)
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        # pooling
        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)

        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self._connector(x)  # (B, L, D)  8*8*4=256

        # scoring
        B, L, D = x.shape
        scores = x.reshape(B, 4, L//4, D)
        scores = scores.repeat_interleave(8, dim=1)  # (B, 32, L//4, D)
        scores = torch.cat([scores, vision2d], dim=2)  # (B, 32, L//4+L, D)
        scores = scores.mean(dim=2)  # (B, 32, D)

        text = self.text_mlp(text)
        scores = torch.bmm(scores, text.unsqueeze(-1))  # (B, 32, 1)
        # scores = scores / torch.sqrt(torch.tensor(D, dtype=scores.dtype, device=scores.device))
        temperature = 5
        scores = scores / temperature  # (B, 32, 1)
        scores = F.softmax(scores, dim=1)
        scores = self.attn_dropout(scores)
        print("scores:", scores.shape, scores[0].squeeze())

        return x, scores
