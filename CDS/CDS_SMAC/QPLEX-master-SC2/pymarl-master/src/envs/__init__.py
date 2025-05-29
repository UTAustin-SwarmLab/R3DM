from functools import partial

import sys
import os


def env_fn(env_type, **kwargs):
    if env_type== "sc2":
        from smac.env import MultiAgentEnv, StarCraft2Env
        return StarCraft2Env(**kwargs)
    elif env_type == "sc2v2":
        from .smacv2_env import SMACv2Env
        return SMACv2Env(kwargs)
    else:
        raise ValueError("env {} is not registered".format(env_type))
    


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
