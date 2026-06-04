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

NAME=tmp_8
NUM_GPUS=4
MASTER_PORT=18920       
CONFIG_NAME="Qwen3.5-9B"       
SOURCE=grouptransmla
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=False
RESUME_GLOBAL_STEP=latest   # -1: don't resume,   int: resume from global steps,  latest: resume from latest
LEARNING_RATE=5e-5
CONVERSATION_MAX_LEN=1270   # Extra base len: 0 Extra chat len per turn: 10
CONTEXT_MAX_LEN=$((CONVERSATION_MAX_LEN - 9)) # $((CONVERSATION_MAX_LEN - 10))
TYPE=transformer
NUM_LAYERS=4
WARMUP_STEPS=200
METHOD=rl
LORA_R=8
METALORA_R=8

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
LORA_SCOPE=attention   # attention (q/k/v/o on full-attn layers) | all
MC_ARGS="model.lora_scope=$LORA_SCOPE multichunk.enabled=$MULTICHUNK_ENABLED multichunk.n_sink=$N_SINK multichunk.n_local=$N_LOCAL multichunk.page_size=$PAGE_SIZE multichunk.query_aware_mix=$QUERY_AWARE_MIX multichunk.mix_top_k=$MIX_TOP_K multichunk.mix_temp=$MIX_TEMP multichunk.norm_rule=$NORM_RULE multichunk.norm_lam=$NORM_LAM multichunk.var_rank=$VAR_RANK multichunk.mix_rescale=$MIX_RESCALE"

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    ((MASTER_PORT++))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO

python generate_group_idx.py  \
    --config-name $CONFIG_NAME \
    name=$NAME \
    mode=pretrain \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    resume_global_step=$RESUME_GLOBAL_STEP \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    data.context_max_length=$CONTEXT_MAX_LEN \
    metanetwork.linear_gate_cfg.num_layers=$NUM_LAYERS \
    optim.warmup_steps=$WARMUP_STEPS \
    metanetwork.method=$METHOD \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    ${MC_ARGS} \
    > tmp_pretrain_$NAME.txt 2>&1

wait

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    meta_train_parallel.py \
    --config-name $CONFIG_NAME \
    name=$NAME \
    mode=pretrain \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    resume_global_step=$RESUME_GLOBAL_STEP \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    data.context_max_length=$CONTEXT_MAX_LEN \
    metanetwork.linear_gate_cfg.num_layers=$NUM_LAYERS \
    optim.warmup_steps=$WARMUP_STEPS \
    metanetwork.method=$METHOD \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    ${MC_ARGS} \
    > tmp_pretrain_$NAME.txt 2>&1 &
