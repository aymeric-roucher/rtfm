import os
from typing import Union

try:
    from llama_recipes.model_checkpointing.checkpoint_handler import (
        fullstate_save_policy as _fullstate_save_policy,
    )
    _LLAMA_RECIPES_AVAILABLE = True
except Exception:
    _LLAMA_RECIPES_AVAILABLE = False

from rtfm.configs import TrainConfig
from torch.distributed import fsdp as FSDP
from torch.distributed.fsdp import StateDictType

if _LLAMA_RECIPES_AVAILABLE:
    fullstate_save_policy = _fullstate_save_policy
else:
    def fullstate_save_policy(*args, **kwargs):
        return None


def fetch_auth_token() -> Union[str, None]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
