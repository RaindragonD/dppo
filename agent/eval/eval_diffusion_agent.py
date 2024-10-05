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

def generate_html(render_dir, n_render, episodes_start_end, reward_trajs_split):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Videos</title>
        <style>
            .video-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 20px;
            }
            .video-container {
                border: 5px solid;
                border-radius: 10px;
                overflow: hidden;
            }
            .success { border-color: #4CAF50; }
            .failure { border-color: #F44336; }
            video {
                width: 100%;
                display: block;
            }
            .video-info {
                padding: 10px;
                text-align: center;
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        <div class="video-grid">
    """

    # Create a dictionary to store episodes for each env_id
    env_episodes = {}
    for i, (env_id, _, _) in enumerate(episodes_start_end):
        if env_id not in env_episodes:
            env_episodes[env_id] = []
        env_episodes[env_id].append(i)

    for env_id in range(n_render):
        if env_id in env_episodes:
            for itr, episode_index in enumerate(env_episodes[env_id]):
                video_path = f"trial-{env_id}_reset-{itr}.mp4"
                reward_traj = reward_trajs_split[episode_index]
                success = np.max(reward_traj) >= 1
                status_class = "success" if success else "failure"
                html_content += f"""
                    <div class="video-container {status_class}">
                        <video controls>
                            <source src="{html.escape(video_path)}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div class="video-info">
                            Env ID: {env_id}, Episode: {itr}<br>
                            Traj Length: {len(reward_traj)}<br>
                            Max Reward: {np.max(reward_traj):.2f}
                        </div>
                    </div>
                """

    html_content += """
        </div>
    </body>
    </html>
    """

    html_file_path = os.path.join(render_dir, "evaluation_videos.html")
    with open(html_file_path, "w") as f:
        f.write(html_content)

    log.info(f"HTML file with video grid created at: {html_file_path}")

class EvalDiffusionAgent(EvalAgent):

    def __init__(self, cfg, load_model=True):
        super().__init__(cfg, load_model)

    def decode_predictions(self, x):
        action_dim = self.cfg.action_dim
        horizon_steps = self.cfg.horizon_steps
        B = x.shape[0]
        if self.cfg.predict_state:
            states = x[:, action_dim * horizon_steps:]
            actions = x[:, :action_dim * horizon_steps].reshape(B, horizon_steps, action_dim)
            return actions, states
        else:
            actions = x.reshape(B, horizon_steps, action_dim)
            return actions, None
        
    def run(self):

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        for env_ind in range(self.n_render):
            options_venv[env_ind]["video_path"] = os.path.join(
                self.render_dir, f"trial-{env_ind}.mp4"
            )
        # init_states = np.load("log/robomimic-eval/init_states.npy")[[11,13,15,17],0]
        # for env_ind in range(len(init_states)):
        #     options_venv[env_ind]["init_state"] = init_states[env_ind]

        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.empty((0, self.n_envs))

        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps)):

            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = self.model(cond=cond, deterministic=True)
                actions, _ = self.decode_predictions(samples.trajectories.cpu().numpy())
            action_venv = actions[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, done_venv, info_venv = self.venv.step(action_venv)
            if step == 0:
                sim_states = np.array([info["states"] for info in info_venv])
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
            # if (
            #     self.furniture_sparse_reward
            # ):  # only for furniture tasks, where reward only occurs in one env step
            #     episode_best_reward = episode_reward
            # else:
            #     episode_best_reward = np.array(
            #         [
            #             np.max(reward_traj) / self.act_steps
            #             for reward_traj in reward_trajs_split
            #         ]
            #     )
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

        # After the evaluation loop, call the generate_html function
        generate_html(self.render_dir, self.n_render, episodes_start_end, reward_trajs_split)

        return success_rate
