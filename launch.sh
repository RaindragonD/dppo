python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/square
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/dynamics

python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/robomimic/pretrain/square
python script/run.py --config-name=pre_diffusion_state_mlp --config-dir=cfg/robomimic/pretrain/square

python script/run.py --config-name=pre_critic --config-dir=cfg/robomimic/pretrain/square

python script/run.py --config-name=ft_ppo_diffusion_mlp --config-dir=cfg/robomimic/finetune/square

python script/run.py --config-name=ft_diffusion_mlp --config-dir=cfg/dynamics
python script/run.py --config-name=pre_dynamics_mlp --config-dir=cfg/dynamics
python script/run.py --config-name=pre_inverse_dynamics_mlp --config-dir=cfg/dynamics

python script/dataset/process_robomimic_dataset.py --load_path ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 --save_dir data/robomimic/square-img-ph --cameras agentview --normalize
python script/dataset/process_robomimic_dataset.py --load_path ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 --save_dir data/robomimic/square-ph --normalize

python script/dataset/process_robomimic_dataset_multi.py --load_path ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 --save_dir data/robomimic/square-ph --normalize
python script/dataset/process_robomimic_dataset_multi.py --load_paths ~/bc/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5 ~/bc/diffusion_policy/data/robomimic/datasets/can/ph/image.hdf5 ~/bc/diffusion_policy/data/robomimic/datasets/lift/ph/image.hdf5 --save_dir data/robomimic/multi-ph --normalize

# collect dynamics data
python script/run.py --config-name=collect_dynamics_data --config-dir=cfg/dynamics