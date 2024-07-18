import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class WaveNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass