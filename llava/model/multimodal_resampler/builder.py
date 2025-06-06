import torch

from .sem_perceiver import Semantic_Perceiver

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, **kwargs):
    # resampler_type = getattr(model_args, "mm_resampler_type", None)
    resampler_type = "semantic_perceiver"
    if resampler_type == "semantic_perceiver":
        return Semantic_Perceiver(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()

    raise ValueError(f"Unknown resampler type: {resampler_type}")
