1. python eval_rollout.py   --ckpt checkpoints/best.pt   --task push-v3   --episodes 10   --max-steps 100   --record-video Output/SigLip2/siglip2_30/push-v3.mp4

2. python viz_actions.py   --ckpt checkpoints/best.pt   --config experiments/siglip2_config.yaml   --data-root data/short-metaworld-vla   --task door-open-v3   --out-dir plots_door_open_v3
