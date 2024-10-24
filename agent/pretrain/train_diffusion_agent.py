"""
Pre-training diffusion policy

"""

import logging
import wandb
import numpy as np

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device
from agent.eval.eval_diffusion_agent import EvalDiffusionAgent

import torch

class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sample_freq = cfg.train.get("sample_freq", 10)
        self.eval_freq = cfg.train.get("eval_freq", 50)
        
        # build eval agent
        self.eval_agent = EvalDiffusionAgent(cfg, load_model=False)
        self.eval_agent.model = self.model

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

        timer = Timer()
        self.epoch = 0
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
            
            # sample actions
            if self.epoch % self.sample_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    for batch_train in self.dataloader_train:
                        gt, cond = batch_train
                        pred = self.model(cond=cond).trajectories
                        gt_actions, gt_states = self.decode_predictions(gt)
                        pred_actions, pred_states = self.decode_predictions(pred)
                        action_mse = torch.nn.functional.mse_loss(pred_actions, gt_actions)
                        wandb.log({
                            "train - action mse": action_mse,
                        }, step=self.epoch)
                        if self.cfg.predict_state:
                            state_mse = torch.nn.functional.mse_loss(pred_states, gt_states)
                            wandb.log({
                                "train - state mse": state_mse,
                            }, step=self.epoch)
                        
                self.model.train()
                
            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log({"loss - train": loss_train}, step=self.epoch)
                    
            # eval
            if self.epoch % self.eval_freq == 0:
                self.model.eval()
                success_rate = self.eval_agent.run()
                wandb.log({"eval - success rate": success_rate,}, step=self.epoch)
                self.model.train()
                
            # count
            self.epoch += 1
