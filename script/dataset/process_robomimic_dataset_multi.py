"""
Process robomimic dataset and save it into our custom format so it can be loaded for diffusion training.

Using some code from robomimic/robomimic/scripts/get_dataset_info.py

can-mh:
    total transitions: 62756
    total trajectories: 300
    traj length mean: 209.18666666666667
    traj length std: 114.42181532479817
    traj length min: 98
    traj length max: 1050
    action min: -1.0
    action max: 1.0

    {
        "env_name": "PickPlaceCan",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "has_renderer": false,
            "has_offscreen_renderer": false,
            "ignore_done": true,
            "use_object_obs": true,
            "use_camera_obs": false,
            "control_freq": 20,
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [
                    0.05,
                    0.05,
                    0.05,
                    0.5,
                    0.5,
                    0.5
                ],
                "output_min": [
                    -0.05,
                    -0.05,
                    -0.05,
                    -0.5,
                    -0.5,
                    -0.5
                ],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [
                    0,
                    300
                ],
                "damping_limits": [
                    0,
                    10
                ],
                "position_limits": null,
                "orientation_limits": null,
                "uncouple_pos_ori": true,
                "control_delta": true,
                "interpolation": null,
                "ramp_ratio": 0.2
            },
            "robots": [
                "Panda"
            ],
            "camera_depths": false,
            "camera_heights": 84,
            "camera_widths": 84,
            "reward_shaping": false
        }
    }

robomimic dataset normalizes action to [-1, 1], observation roughly? to [-1, 1]. Seems sometimes the upper value is a bit larger than 1 (but within 1.1).

"""

import numpy as np
from tqdm import tqdm
import pickle
import cv2

try:
    import h5py  # not included in pyproject.toml
except:
    print("Installing h5py")
    os.system("pip install h5py")
import os
import random
from copy import deepcopy
import logging


def make_dataset(
    load_paths,
    save_dir,
    save_name_prefix,
    val_split,
    normalize,
):
    # get basic stats
    traj_lengths = []
    all_actions = []
    all_obs = []

    low_dim_obs_names = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    if args.cameras is None:  # state-only
        low_dim_obs_names.append("object")
            
    # Load hdf5 file from load_path
    for load_path in load_paths:
        with h5py.File(load_path, "r") as f:
            # put demonstration list in increasing episode order
            demos = sorted(list(f["data"].keys()))
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]
            
            for ep in demos:
                traj_lengths.append(f[f"data/{ep}/actions"].shape[0])
                obs = np.hstack(
                    [
                        f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                        for low_dim_obs_name in low_dim_obs_names
                    ]
                )
                actions = f[f"data/{ep}/actions"][:]
                all_actions.append(actions)
                if obs.shape[1] == 19:
                    dummy_quat = np.zeros((obs.shape[0], 4))
                    dummy_quat[:, 3] = 1
                    obs = np.hstack([obs, dummy_quat])
                all_obs.append(obs)


    # split indices in train and val
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    train_indices = random.sample(range(num_traj), k=num_train)

    # do over all indices
    out_train = {"states": [], "actions": [], "traj_lengths": []}
    out_val = deepcopy(out_train)
    
    # import ipdb; ipdb.set_trace()
    obs_min = np.min(np.concatenate(all_obs, axis=0), axis=0)
    obs_max = np.max(np.concatenate(all_obs, axis=0), axis=0)
    action_min = np.min(np.concatenate(all_actions, axis=0), axis=0)
    action_max = np.max(np.concatenate(all_actions, axis=0), axis=0)
    
    for i in tqdm(range(num_traj)):
        if i in train_indices:
            out = out_train
        else:
            out = out_val
            
        raw_obs = all_obs[i]
        raw_actions = all_actions[i]
        
        # scale to [-1, 1] for both ob and action
        if normalize:
            def normalize_fn(raw, min, max):
                return 2 * (raw - min) / (max - min + 1e-6) - 1
            obs = normalize_fn(raw_obs, obs_min, obs_max)
            actions = normalize_fn(raw_actions, action_min, action_max)
        else:
            obs = raw_obs
            actions = raw_actions

        data_traj = {
            "states": obs,
            "actions": actions,
            "traj_lengths": traj_lengths[i],
        }
        
        # apply padding to make all episodes have the same max steps
        # later when we load this dataset, we will use the traj_length to slice the data
        for key in data_traj.keys():
            out[key].append(data_traj[key])
    
    out_train["states"] = np.concatenate(out_train["states"], axis=0)
    out_train["actions"] = np.concatenate(out_train["actions"], axis=0)
    out_train["traj_lengths"] = np.array(out_train["traj_lengths"])
    out_val["states"] = np.concatenate(out_val["states"], axis=0)
    out_val["actions"] = np.concatenate(out_val["actions"], axis=0)
    out_val["traj_lengths"] = np.array(out_val["traj_lengths"])
    
    # report statistics on the data
    logging.info("===== Basic stats =====")
    traj_lengths = np.array(traj_lengths)
    logging.info("total transitions: {}".format(np.sum(traj_lengths)))
    logging.info("total trajectories: {}".format(traj_lengths.shape[0]))
    logging.info(
        f"traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
    )
    logging.info(
        f"traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
    )
    logging.info(f"obs min: {obs_min}")
    logging.info(f"obs max: {obs_max}")
    logging.info(f"action min: {action_min}")
    logging.info(f"action max: {action_max}")
    
    # Save to np file
    save_train_path = os.path.join(save_dir, save_name_prefix + "train.npz")
    save_val_path = os.path.join(save_dir, save_name_prefix + "val.npz")
    with open(save_train_path, "wb") as f:
        pickle.dump(out_train, f)
    with open(save_val_path, "wb") as f:
        pickle.dump(out_val, f)
    if normalize:
        normalization_save_path = os.path.join(
            save_dir, save_name_prefix + "normalization.npz"
        )
        np.savez(
            normalization_save_path,
            obs_min=obs_min,
            obs_max=obs_max,
            action_min=action_min,
            action_max=action_max,
        )

    # debug
    logging.info("\n========== Final ===========")
    logging.info(
        f"Train - Number of episodes and transitions: {len(out_train['traj_lengths'])}, {np.sum(out_train['traj_lengths'])}"
    )
    logging.info(
        f"Val - Number of episodes and transitions: {len(out_val['traj_lengths'])}, {np.sum(out_val['traj_lengths'])}"
    )
    logging.info(
        f"Train - Mean/Std trajectory length: {np.mean(out_train['traj_lengths'])}, {np.std(out_train['traj_lengths'])}"
    )
    logging.info(
        f"Train - Max/Min trajectory length: {np.max(out_train['traj_lengths'])}, {np.min(out_train['traj_lengths'])}"
    )
    if val_split > 0:
        logging.info(
            f"Val - Mean/Std trajectory length: {np.mean(out_val['traj_lengths'])}, {np.std(out_val['traj_lengths'])}"
        )
        logging.info(
            f"Val - Max/Min trajectory length: {np.max(out_val['traj_lengths'])}, {np.min(out_val['traj_lengths'])}"
        )
    obs_dim = out_train["states"].shape[1]
    action_dim = out_train["actions"].shape[1]
    for obs_dim_ind in range(obs_dim):
        obs = out_train["states"][:, obs_dim_ind]
        logging.info(
            f"Train - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
        )
    for action_dim_ind in range(action_dim):
        action = out_train["actions"][:, action_dim_ind]
        logging.info(
            f"Train - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
        )
    if val_split > 0:
        for obs_dim_ind in range(obs_dim):
            obs = out_val["states"][:, obs_dim_ind]
            logging.info(
                f"Val - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
            )
        for action_dim_ind in range(action_dim):
            action = out_val["actions"][:, action_dim_ind]
            logging.info(
                f"Val - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
            )
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_paths", type=str, nargs="+")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default="0.02")
    parser.add_argument("--max_episodes", type=int, default="-1")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cameras", nargs="*", default=None)
    args = parser.parse_args()

    import datetime

    if args.max_episodes > 0:
        args.save_name_prefix += f"max_episodes_{args.max_episodes}_"

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    make_dataset(
        args.load_paths,
        args.save_dir,
        args.save_name_prefix,
        args.val_split,
        args.normalize,
    )