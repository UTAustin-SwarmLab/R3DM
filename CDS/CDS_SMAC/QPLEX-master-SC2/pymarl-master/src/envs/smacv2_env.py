from __future__ import absolute_import, division, print_function

import time
from os import replace

import numpy as np
from absl import logging
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)
import os.path as osp
from pathlib import Path
import yaml

from gym.spaces import Box, Discrete


class SMACv2Env:
    def __init__(self, args):
        self.map_config = args
        self.env = StarCraftCapabilityEnvWrapper(**self.map_config)
        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.timeouts = self.env.env.timeouts

        self.share_observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(state_shape,))
        )
        self.observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        )
        self.action_space = self.repeat(Discrete(n_actions))
        self.episode_limit = self.env.episode_limit
        self.shield_bits_ally = self.env.shield_bits_ally
        self.unit_type_bits = self.env.unit_type_bits
    # def step(self, actions):
    #     processed_actions = np.squeeze(actions, axis=1).tolist()
    #     reward, terminated, info = self.env.step(actions)
    #     # obs = self.env.get_obs()
    #     # state = self.repeat(self.env.get_state())
    #     # rewards = [[reward]] * self.n_agents
    #     # dones = [terminated] * self.n_agents
    #     # if terminated:
    #     #     if self.env.env.timeouts > self.timeouts:
    #     #         assert (
    #     #             self.env.env.timeouts - self.timeouts == 1
    #     #         ), "Change of timeouts unexpected."
    #     #         info["bad_transition"] = True
    #     #         self.timeouts = self.env.env.timeouts
    #     # infos = [info] * self.n_agents
    #     # avail_actions = self.env.get_avail_actions()
    #     return obs, state, rewards, dones, infos, avail_actions

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        state = self.repeat(self.env.get_state())
        avail_actions = self.env.get_avail_actions()
        return obs, state, avail_actions


    def close(self):
        self.env.close()

    def load_map_config(self, map_name):
        base_path = osp.split(osp.split(osp.dirname(osp.abspath(__file__)))[0])[0]
        map_config_path = (
            Path(base_path)
            / "configs"
            / "env_configs"
            / "smacv2"
            / f"{map_name}.yaml"
        )
        with open(str(map_config_path), "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)
        return map_config

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def get_observation_space(self):
        return self.observation_space
    
    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()
