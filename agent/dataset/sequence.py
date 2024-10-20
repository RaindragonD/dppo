"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "target cond")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        predict_state=False,
        device="cuda:0",
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.predict_state = predict_state
        self.device = device
        self.use_img = use_img

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        self.traj_lengths = traj_lengths

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, obs_dim)
        self.actions = (
            torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                device
            )  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

    def get_actions_states(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : end]
        actions = self.actions[start:end]
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        
        end_state = self.states[end]
        return actions, conditions, end_state
    
    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        actions, conditions, end_state = self.get_actions_states(idx)

        actions_flattened = actions.view(-1)
        if self.predict_state:
            actions_state = torch.cat([actions_flattened, end_state], dim=-1)
            batch = Batch(actions_state, conditions)
        else:
            batch = Batch(actions_flattened, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps #  NOTE: keep last observation, but then last action is not used
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)

Batch_Dynamics = namedtuple("Batch_Dynamics", "states actions end_state")

class DynamicsDataset(StitchedSequenceDataset):
    
    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        actions, conditions, end_state = self.get_actions_states(idx)
        cond = torch.cat([conditions["state"].view(-1), actions.view(-1)], dim=0)
        batch = Batch(end_state, cond)
        return batch
    
class InverseDynamicsDataset(StitchedSequenceDataset):
    
    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        actions, conditions, end_state = self.get_actions_states(idx)
        cond = torch.cat([conditions["state"].view(-1), end_state.view(-1)], dim=0)
        batch = Batch(actions.view(-1), cond)
        return batch

class DynamicsDatasetOnline(DynamicsDataset):
    
    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        predict_state=False,
        device="cuda:0",
        normalization_path=None,
    ):
        super().__init__(dataset_path, horizon_steps, cond_steps, img_cond_steps, max_n_episodes, use_img, predict_state, device)
        self.normalize_obs = False
        if normalization_path is not None:
            self.normalization_data = np.load(normalization_path)
            self.normalize_obs = True
    
    def normalize_data(self, states, actions):

        obs_min = self.normalization_data['obs_min']
        obs_max = self.normalization_data['obs_max']
        action_min = self.normalization_data['action_min']
        action_max = self.normalization_data['action_max']
        
        states = (states - obs_min) / (obs_max - obs_min)
        actions = (actions - action_min) / (action_max - action_min)
        
        return states, actions
    
    def add_trajectory(self, states, actions):
        assert states.shape[0] == actions.shape[0], f"states and actions must have the same length, found {states.shape[0]} and {actions.shape[0]}"
        if self.normalize_obs:
            states, actions = self.normalize_data(states, actions)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        
        cur_traj_index = len(self.states)
        traj_length = states.shape[0]
        max_start = cur_traj_index + traj_length - self.horizon_steps #  NOTE: keep last observation, but then last action is not used
        self.indices += [
            (i, i - cur_traj_index) for i in range(cur_traj_index, max_start)
        ]
        
        self.states = torch.cat((self.states, states), dim=0)
        self.actions = torch.cat((self.actions, actions), dim=0)
        self.traj_lengths = np.concatenate((self.traj_lengths, [traj_length]))

class CriticDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path,
        cond_steps=1,
        act_steps=1,
        device="cuda:0",
        max_n_episodes=10000
    ):
        
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.device = device

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array

        self.states = torch.empty((0,)).to(self.device)
        self.dones_trajs = torch.empty((0,)).to(self.device)
        self.firsts_trajs = torch.empty((0,)).to(self.device)
        cur_traj_index = 0
        for traj_length in traj_lengths:
            states = dataset["states"][cur_traj_index:cur_traj_index+traj_length]
            indices = list(range(0, len(states), 8))
            if indices[-1] != len(states) - 1:
                indices.append(len(states) - 1)
            states_subsampled = torch.from_numpy(states[indices]).to(self.device)
            dones = torch.zeros((states_subsampled.shape[0])).to(self.device)
            firsts = torch.zeros((states_subsampled.shape[0])).to(self.device)
            firsts[0] = 1
            dones[-1] = 1
            
            self.states = torch.cat((self.states, states_subsampled), dim=0).float()
            self.dones_trajs = torch.cat((self.dones_trajs, dones), dim=0)
            self.firsts_trajs = torch.cat((self.firsts_trajs, firsts), dim=0)
            cur_traj_index += traj_length

        self.rewards_trajs = self.dones_trajs.clone()
        
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")

    def __getitem__(self, idx):
        
        states = torch.stack(
            [
                self.states[max(idx - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        
        done = self.dones_trajs[idx]
        reward = self.rewards_trajs[idx]
        first = self.firsts_trajs[idx]
        
        return conditions, reward, done, first

    def __len__(self):
        return len(self.states)
