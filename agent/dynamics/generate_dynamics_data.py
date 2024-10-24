import numpy as np
import h5py
import os
from agent.dynamics.vector_env import VectorEnv

class GenerateDynamicsDataAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_envs = cfg.env.n_envs
        self.act_steps = cfg.act_steps
        self.n_steps = 100
        self.n_render = cfg.render_num
        self.n_render = min(self.n_render, self.n_envs)
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.action_dim = cfg.action_dim
        
        self.venv = VectorEnv(cfg)
        self.all_states = []
        self.all_actions = []

        hdf5_path = "/home/yl/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5"
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            data = hdf5_file['data']
            for demo_name in data.keys():
                demo = data[demo_name]
                states = demo["states"][:]
                actions = demo["actions"][:]
                self.all_states.append(states)
                self.all_actions.append(actions)
                self.n_steps = min(self.n_steps, len(actions))
        
        normalization_path = "data/robomimic/square-ph/normalization.npz"
        normalization = np.load(normalization_path)
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]
    
    def normalize_action(self, action):
        action = 2 * (
            (action - self.action_min) / (self.action_max - self.action_min + 1e-6) - 0.5
        )
        return action

    def init_envs(self, init_states=None):
        options_venv = [{} for _ in range(self.n_envs)]
        for env_ind in range(self.n_render):
            options_venv[env_ind]["video_path"] = os.path.join(
                self.render_dir, f"trial-{env_ind}.mp4"
            )
        if init_states is not None:
            for env_ind in range(len(init_states)):
                options_venv[env_ind]["init_state"] = init_states[env_ind]
        obs = self.venv.reset_env_all(options_venv=options_venv)
        return obs
    
    def run(self):
        
        # sample object / robot start configs
        init_states = []
        actions = []

        for _ in range(self.n_envs):
            state_demo_id = np.random.randint(0, len(self.all_states))
            state_id = np.random.randint(0, len(self.all_states[state_demo_id]))
            state = self.all_states[state_demo_id][state_id]
            init_states.append(state)
            
            if np.random.rand() < 0.5:
                n_action_targets = 5
                action_targets = np.zeros((n_action_targets, self.action_dim))
                for i in range(n_action_targets):
                    action_targets[i, 0:3] = np.random.uniform(-0.2, 0.2, 3)
                    action_targets[i, 3:6] = np.random.uniform(-0.2, 0.2, 3)
                    action_targets[i, 6] = np.random.choice([-1, 1])
                interpolated_actions = np.zeros((100, self.action_dim))
                for i in range(self.action_dim - 1):  # Interpolate all dimensions except the last
                    interpolated_actions[:, i] = np.interp(
                        np.linspace(0, n_action_targets-1, 100),
                        np.arange(n_action_targets),
                        action_targets[:, i]
                    )
                # For the last dimension, copy the closest action target
                closest_indices = np.round(np.linspace(0, n_action_targets - 1, 100)).astype(int)
                interpolated_actions[:, -1] = action_targets[closest_indices, -1]

                actions.append(interpolated_actions)
            else:
                action_demo_id = np.random.randint(0, len(self.all_actions))
                action_start_id = np.random.randint(0, len(self.all_actions[action_demo_id])-self.n_steps)
                actions.append(self.all_actions[action_demo_id][action_start_id:action_start_id+self.n_steps])
        
        obs = self.init_envs(init_states=init_states)
        actions = np.array(actions)
        normalized_actions = self.normalize_action(actions)

        # sample action sequences
        for step in range(self.n_steps):
            step_actions = normalized_actions[:, step:step+1, :]
            obs, reward, done, info = self.venv.step(step_actions)
        