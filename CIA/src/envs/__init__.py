from functools import partial
from .multiagentenv import MultiAgentEnv


import sys
import os

def env_fn(env_type, **kwargs) -> MultiAgentEnv:
    
    if env_type == "sc2":
        from .starcraft2 import StarCraft2Env
        return StarCraft2Env(**kwargs)
    elif env_type == "turn":
        from .turn import TurnEnv
        return TurnEnv(**kwargs)
    elif env_type == "sc2v2":
        from .smacv2_env import SMACv2Env
        return SMACv2Env(kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["turn"] = partial(env_fn, env=TurnEnv)
