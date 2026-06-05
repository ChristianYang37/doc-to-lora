#!/bin/bash

#SBATCH -J test
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o test.out
#SBATCH -e test.err

NAME=8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150
NUM_GPUS=1
MASTER_PORT=18900             
CONFIG_NAME="Qwen3-8B"       
TEST_BATCH_SIZE=4
TEST_GLOBAL_STEP=20000
TEST_SOURCE=msmarco_v2 # squad hotpotqa 2wikimultihopqa musique msmarco_v1 msmarco_v2
NUM_LAYERS=4
METHOD=rl
CONTEXT_AVG_LEN=2048
CONTEXT_MAX_LENGTH=4500
CONVERSATION_MAX_LENGTH=300
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
LORA_ORTHO_UPDATE=true
ORTHO_COMBINE=true
MC_ARGS="multichunk.enabled=$MULTICHUNK_ENABLED multichunk.n_sink=$N_SINK multichunk.n_local=$N_LOCAL multichunk.page_size=$PAGE_SIZE multichunk.query_aware_mix=$QUERY_AWARE_MIX multichunk.mix_top_k=$MIX_TOP_K multichunk.mix_temp=$MIX_TEMP multichunk.norm_rule=$NORM_RULE multichunk.norm_lam=$NORM_LAM multichunk.var_rank=$VAR_RANK multichunk.mix_rescale=$MIX_RESCALE lora_ortho_update=$LORA_ORTHO_UPDATE multichunk.ortho_combine=$ORTHO_COMBINE"

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

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    test.py \
    --config-name $CONFIG_NAME \
    name=$NAME \
    test.batch_size=$TEST_BATCH_SIZE \
    test_global_step=$TEST_GLOBAL_STEP \
    test.source=$TEST_SOURCE \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    metanetwork.method=$METHOD \
    test.context_avg_len=$CONTEXT_AVG_LEN \
    model.lora_r=$LORA_R \
    model.metalora_r=$METALORA_R \
    test.context_max_length=$CONTEXT_MAX_LENGTH \
    test.conversation_max_length=$CONVERSATION_MAX_LENGTH \
    ${MC_ARGS} \
    > tmp_test_${TEST_SOURCE}_$NAME.txt 2>&1 &
