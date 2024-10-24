"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.

"""

import os
import numpy as np
import torch
import logging
import html

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from tqdm import tqdm
import h5py

class EvalInverseDynamicsAgent(EvalAgent):

    def __init__(self, cfg, load_model=True):
        super().__init__(cfg, load_model)
        self.init_states = []
        self.obs_states = []
        
        self.normalize = cfg.normalization_path
        normalization = np.load(cfg.normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]
            
        hdf5_path = "/home/yl/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5"
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            data = hdf5_file['data']
            for demo_name in data.keys():
                demo = data[demo_name]
                states = demo["states"][0]
                self.init_states.append(states)
                
                low_dim_obs_names = [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object",
                ]
                obs = np.hstack([demo["obs"][low_dim_obs_name][:] for low_dim_obs_name in low_dim_obs_names])
                obs = 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 1
                self.obs_states.append(obs[::self.cfg.act_steps] + [obs[-1]])
        max_length = max(len(obs) for obs in self.obs_states)
        for i, obs in enumerate(self.obs_states):
            self.obs_states[i] = np.concatenate([obs, np.repeat(obs[-1][np.newaxis, :], max_length - len(obs), axis=0)], axis=0)
        self.obs_states = np.array(self.obs_states)
                
    def decode_predictions(self, x):
        action_dim = self.cfg.action_dim
        horizon_steps = self.cfg.horizon_steps
        obs_dim = self.cfg.obs_dim
        B, L = x.shape
        if L == horizon_steps * action_dim + obs_dim:
            states = x[:, action_dim * horizon_steps:]
            actions = x[:, :action_dim * horizon_steps].reshape(B, horizon_steps, action_dim)
            return actions, states
        elif L == horizon_steps * action_dim:
            actions = x.reshape(B, horizon_steps, action_dim)
            return actions, None
        else:
            raise ValueError(f"Invalid input length: {L}")
        
    def run(self):

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        for env_ind in range(self.n_render):
            options_venv[env_ind]["video_path"] = os.path.join(
                self.render_dir, f"trial-{env_ind}.mp4"
            )
        for env_ind in range(self.n_envs):
            options_venv[env_ind]["init_state"] = self.init_states[env_ind]

        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.empty((0, self.n_envs))

        # Collect a set of trajectories from env
        for step in tqdm(range(self.obs_states.shape[1])):

            # Select action
            with torch.no_grad():
                state = np.concatenate([prev_obs_venv["state"].reshape(self.n_envs, -1), self.obs_states[:self.n_envs, step]], axis=1)
                state = torch.from_numpy(state).float().to(self.device)
                samples = self.model(cond=state, deterministic=True)
                actions, _ = self.decode_predictions(samples.trajectories.cpu().numpy())
            action_venv = actions[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, done_venv, info_venv = self.venv.step(action_venv)
            if step == 0:
                sim_states = np.array([info["env_states"] for info in info_venv])
            reward_trajs = np.vstack((reward_trajs, reward_venv[None]))
            firsts_trajs[step + 1] = done_venv
            prev_obs_venv = obs_venv
        
        np.save(os.path.join(self.logdir, "init_states"), sim_states)

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            episode_best_reward = episode_reward # NOTE: using robomimic only now, sparse reward, reset upon finish
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            reward_trajs_split = []
            log.info("[WARNING] No episode completed within the iteration!")

        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )

        return success_rate
