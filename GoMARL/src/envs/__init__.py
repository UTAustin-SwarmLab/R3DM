from functools import partial
import sys
import os





try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
    from .grf import (Academy_3_vs_1_with_Keeper, 
                      Academy_Counterattack_Easy,
                      Academy_Pass_and_Shoot_with_Keeper)
except:
    gfootball = False

def env_fn(env_type, **kwargs):
    if env_type=='sc2':
        from .multiagentenv import MultiAgentEnv
        from .starcraft import StarCraft2Env
        return StarCraft2Env(**kwargs)
    elif env_type=='gfootball':
        try:
            gfootball = True
            from .gfootball import GoogleFootballEnv
            from .grf import (Academy_3_vs_1_with_Keeper, 
                            Academy_Counterattack_Easy,
                            Academy_Pass_and_Shoot_with_Keeper)
        except:
            raise Exception("gfootball not installed")
    elif env_type=='sc2v2':
        from .smacv2_env import SMACv2Env
        return SMACv2Env(kwargs)


# REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# if gfootball:
#     REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
#     REGISTRY["academy_3_vs_1_with_keeper"] = partial(env_fn, env=Academy_3_vs_1_with_Keeper)
#     REGISTRY["academy_counterattack_easy"] = partial(env_fn, env=Academy_Counterattack_Easy)
#     REGISTRY["academy_pass_and_shoot_with_keeper"] = partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
