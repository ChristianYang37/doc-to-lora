#!/bin/bash

#SBATCH -J metalora
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o metalora.out
#SBATCH -e metalora.err

NAME=8gpu_4lora_4metalora_lr5e-5_grouppretrain_6layer_1270
NUM_GPUS=8
MASTER_PORT=18900             
CONFIG_NAME="Qwen3.5-9B"
NUM_EPOCHS=1
EVAL_STEPS=10 # 625
SAVE_STEPS=10 # 625
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=False
CONTEXT_MAX_LEN=2640
CONVERSATION_MAX_LEN=100
RESUME_GLOBAL_STEP=latest
SOURCE=ift-c1qa
WARMUP_STEPS=400
LEARNING_RATE=2.5e-5
TYPE=transformer
NUM_LAYERS=6
METHOD=rl
IFT_ADDITIONAL_METALORA_R=0

# --- doc2lora x SHINE: query-aware + multi-chunk knobs (MULTICHUNK_ENABLED=false -> plain single-pass SHINE) ---
MULTICHUNK_ENABLED=false
N_SINK=4
N_LOCAL=32
PAGE_SIZE=64
QUERY_AWARE_MIX=true
MIX_TOP_K=8
MIX_TEMP=1.0
NORM_RULE=energy
NORM_LAM=1.0
VAR_RANK=true
MIX_RESCALE=1.0
LORA_ORTHO_UPDATE=true
ORTHO_COMBINE=true
LORA_SCOPE=attention   # attention (q/k/v/o on full-attn layers) | all
MC_ARGS="model.lora_scope=$LORA_SCOPE multichunk.enabled=$MULTICHUNK_ENABLED multichunk.n_sink=$N_SINK multichunk.n_local=$N_LOCAL multichunk.page_size=$PAGE_SIZE multichunk.query_aware_mix=$QUERY_AWARE_MIX multichunk.mix_top_k=$MIX_TOP_K multichunk.mix_temp=$MIX_TEMP multichunk.norm_rule=$NORM_RULE multichunk.norm_lam=$NORM_LAM multichunk.var_rank=$VAR_RANK multichunk.mix_rescale=$MIX_RESCALE lora_ortho_update=$LORA_ORTHO_UPDATE multichunk.ortho_combine=$ORTHO_COMBINE"

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    MASTER_PORT=$((MASTER_PORT + 1))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    meta_train_parallel.py \
    --config-name $CONFIG_NAME \
    name=$NAME \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    optim.num_epochs=$NUM_EPOCHS \
    eval.eval_steps=$EVAL_STEPS \
    save.save_steps=$SAVE_STEPS \
    data.context_max_length=$CONTEXT_MAX_LEN \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    resume_global_step=$RESUME_GLOBAL_STEP \
    data.source=$SOURCE \
    optim.warmup_steps=$WARMUP_STEPS \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    metanetwork.method=$METHOD \
    model.ift_additional_metalora_r=$IFT_ADDITIONAL_METALORA_R \
    ${MC_ARGS} \
    > tmp_metatrain_$NAME.txt 2>&1 &
