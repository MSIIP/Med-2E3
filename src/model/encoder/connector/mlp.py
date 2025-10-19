import re

import torch.nn as nn

from src.model.encoder.connector.base import Connector


ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLPConnector(Connector):
    def __init__(self, connector_config):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", connector_config.connector_type)
        act_type = connector_config.connector_type.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(connector_config.input_dim, connector_config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(connector_config.output_dim, connector_config.output_dim)
            )

        self._connector = nn.Sequential(*modules)
