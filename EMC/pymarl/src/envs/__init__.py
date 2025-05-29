from functools import partial
# do not import SC2 in labtop
import socket
import sys
import os
# from .stag_hunt import StagHunt
# from .GridworldEnv import GridworldEnv


def env_fn(env_type, **kwargs):
    if env_type== "sc2":
        if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname():
            
            from smac.env import MultiAgentEnv, StarCraft2Env
            return StarCraft2Env(**kwargs)
        else:
            raise Exception("StarCraft2Env not installed")
        
    elif env_type == "sc2v2":
        from .smacv2_env import SMACv2Env
        return SMACv2Env(kwargs)
    else:
        raise ValueError("env {} is not registered".format(env_type))

# REGISTRY = {
#     "sc2": partial(env_fn, env=StarCraft2Env),
# } if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname() else {}

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
