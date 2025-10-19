import os.path as osp

import torch
import torch.nn as nn


def get_w(weights, keyword):
    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}


class Connector(nn.Module):
    def __init__(self, connector_config=None):
        super().__init__()
        self._connector = None

    def load_model(self, connector_path):
        if connector_path is not None:
            connector_weights = torch.load(
                osp.join(connector_path, "pytorch_model.bin"),
                map_location="cpu",
            )
            self._connector.load_state_dict(get_w(connector_weights, "_connector"))

        for p in self._connector.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self._connector(x)
