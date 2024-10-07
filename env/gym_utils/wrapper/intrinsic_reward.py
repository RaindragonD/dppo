"""
Multi-step wrapper. Allow executing multiple environmnt steps. Returns stacked observation and optionally stacked previous action.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

TODO: allow cond_steps != img_cond_steps (should be implemented in training scripts, not here)
"""

import gym
from typing import Optional
from gym import spaces
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import copy


def quaternion_to_angle(q1, q2):
    dot_product = np.sum(q1 * q2, axis=-1)
    angle = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angle

def compute_pose_diff(pose1, pose2):
    position_diff = np.linalg.norm(pose1[:3] - pose2[:3], axis=-1)
    quaternion_diff = quaternion_to_angle(pose1[3:], pose2[3:])
    return position_diff, quaternion_diff

class IntrinsicRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        change_weight=1,
        coverage_weight=1,
        **kwargs,
    ):
        super().__init__(env)
        self.env = env
        self.change_weight = change_weight
        self.coverage_weight = coverage_weight
        self._single_action_space = env.action_space
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.prev_states = []
        self.steps = 0

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: dict = {},
    ):
        obs = self.env.reset(
            seed=seed,
            options=options,
            return_info=return_info,
        )
        self.prev_states = np.array([obs['state']])
        self.steps = 0
        return obs

    def compute_coverage_reward(self, observation, info):
        prev_pos = self.prev_states[:,9:9+3]
        pos = observation['state'][9:9+3]
        dist = np.linalg.norm(prev_pos - pos, axis=-1)
        reward = np.mean(dist)
        return reward

    def compute_change_reward(self, observation, info):
        if self.steps < 5: return 0
        state = observation['state']
        prev_state = self.prev_states[-1]
        object_pose = state[9:9+7]
        prev_object_pose = prev_state[9:9+7]
        object_pos_diff, object_rot_diff = compute_pose_diff(object_pose, prev_object_pose)
        reward = object_pos_diff + object_rot_diff
        return reward
    
    def compute_intrinsic_reward(self, observation, info):
        
        intrinsic_reward = 0
        intrinsic_reward += self.change_weight * self.compute_change_reward(observation, info)
        intrinsic_reward += self.coverage_weight * self.compute_coverage_reward(observation, info)
        
        return intrinsic_reward

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        observation, env_reward, done, info = self.env.step(action)
        intrinsic_reward = self.compute_intrinsic_reward(observation, info)
        reward = env_reward + intrinsic_reward
        info['env_reward'] = env_reward
        info['intrinsic_reward'] = intrinsic_reward
        self.prev_states = np.vstack([self.prev_states, observation['state'][None]])
        self.steps += 1
        return observation, reward, done, info

    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs)


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml")
    cfg.normalization_path = None
    # shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt
    from env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper
    from env.gym_utils.wrapper.robomimic_lowdim import RobomimicLowdimWrapper

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = IntrinsicRewardWrapper(
        env=RobomimicLowdimWrapper(
            env=env,
        ),
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    obs, env_reward, done, info = wrapper.step(np.zeros(7))
    print(obs.keys())
    # img = wrapper.render()
    # wrapper.close()
    # plt.imshow(img)
    # plt.savefig("test.png")
