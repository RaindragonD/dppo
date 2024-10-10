python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/square

python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/robomimic/pretrain/square
python script/run.py --config-name=pre_diffusion_state_mlp --config-dir=cfg/robomimic/pretrain/square

python script/run.py --config-name=pre_critic --config-dir=cfg/robomimic/pretrain/square

python script/run.py --config-name=ft_ppo_diffusion_mlp --config-dir=cfg/robomimic/finetune/square

python script/run.py --config-name=ft_diffusion_mlp --config-dir=cfg/dynamics
python script/run.py --config-name=pre_dynamics_mlp --config-dir=cfg/dynamics

python script/dataset/process_robomimic_dataset.py --load_path ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 --save_dir data/robomimic/square-img-ph --cameras agentview --normalize
python script/dataset/process_robomimic_dataset.py --load_path ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 --save_dir data/robomimic/square-ph --normalize