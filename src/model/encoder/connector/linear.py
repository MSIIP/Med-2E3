import torch.nn as nn

from src.model.encoder.connector.base import Connector


class LinearConnector(Connector):
    def __init__(self, connector_config):
        super().__init__()
        self._connector = nn.Linear(
            connector_config.input_dim, connector_config.output_dim
        )
