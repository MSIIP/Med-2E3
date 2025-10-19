import torch.nn as nn

from src.model.encoder.connector.base import Connector


class IdentityConnector(Connector):
    def __init__(self, connector_config=None):
        super().__init__()
        self._connector = nn.Identity()
