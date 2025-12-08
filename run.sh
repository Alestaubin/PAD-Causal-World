# Train and Eval with inverse dynamics self-supervision on CausalWorld Reaching task, adapting to finger link mass changes
python3 src/train.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode train \
        --batch_size 512 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --train_steps 100000 \
        --critic_lr 1e-4 \
        --critic_beta 0.9 \
        --critic_tau 0.001 \
        --actor_lr 1e-4 \
        --actor_beta 0.9 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/inv/6 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --use_inv \
        --num_layers 4 \
        --num_shared_layers 2 \
        --note "half of the layers are shared between critic and inverse dynamics encoder" 
        
#################################################################
# train and eval without adaptation for comparison
python3 src/train.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode train \
        --batch_size 256 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --train_steps 100000 \
        --critic_lr 3e-4 \
        --critic_beta 0.9 \
        --critic_tau 0.005 \
        --actor_lr 3e-4 \
        --actor_beta 0.9 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/no_inv/2 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --num_layers 4

python3 src/train.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode train \
        --batch_size 512 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --train_steps 100000 \
        --critic_lr 1e-4 \
        --critic_beta 0.9 \
        --critic_tau 0.001 \
        --actor_lr 1e-4 \
        --actor_beta 0.9 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/inv/7 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --use_inv \
        --num_layers 4 \
        --num_shared_layers 1 \
        --note "Only the last layer is shared between critic and inverse dynamics encoder" 

python3 src/eval.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode "finger_link_mass" \
        --batch_size 256 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/inv/5 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --use_inv \
        --seed 1 \
        --num_layers 4 \
        --num_shared_layers 4 \
        --pad_checkpoint best \
        --pad_num_episodes 5

python3 src/eval.py \
    --domain_name "causalworld" \
    --task_name "reaching" \
    --action_repeat 2 \
    --mode "finger_link_mass" \
    --batch_size 256 \
    --encoder_feature_dim 128 \
    --alpha_lr 3e-4 \
    --work_dir logs/causal_world_reaching/no_inv/1 \
    --save_model \
    --obs_type "structured" \
    --save_video \
    --num_layers 4 \
    --pad_checkpoint best \
    --pad_num_episodes 1

python3 src/eval.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode "finger_link_mass" \
        --batch_size 256 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/inv/8 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --use_inv \
        --seed 1 \
        --num_layers 4 \
        --num_shared_layers 1 \
        --pad_checkpoint best \
        --pad_num_episodes 1


python3 src/train.py \
        --domain_name "causalworld" \
        --task_name "reaching" \
        --action_repeat 2 \
        --mode train \
        --batch_size 512 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --train_steps 100000 \
        --critic_lr 1e-4 \
        --critic_beta 0.9 \
        --critic_tau 0.001 \
        --actor_lr 1e-4 \
        --actor_beta 0.9 \
        --encoder_feature_dim 128 \
        --alpha_lr 3e-4 \
        --work_dir logs/causal_world_reaching/no_inv/2 \
        --save_model \
        --obs_type "structured" \
        --save_video \
        --num_layers 4 \
        --note "No auxiliary task"