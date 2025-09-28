#!/bin/bash

# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export WANDB_DISABLED=true
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=4
MASTER_ADDR=localhost
MASTER_PORT=12355
NNODES=1
RANK=0
# unset OMPI_COMM_WORLD_LOCAL_RANK
# RANK=$OMPI_COMM_WORLD_RANK


LLM_VERSION="Qwen/Qwen2.5-3B-Instruct" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="meta/sam2.1_hiera_large"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Train ################
# setting
PROMPT_VERSION="qwen_2"
RUN_NAME="PAM_ft_${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}" 
echo "MID_RUN_NAME: ${RUN_NAME}"

# CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
CKPT_PATH=models--Qwen--Qwen2.5-3B-Instruct

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path="configs/finetune.yaml" \
    --mm_tunable_parts="mm_vision_resampler,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "/path/to/output/ckpt/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 12288 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    2>&1 | tee /path/to/output/log/train_log-$(date +"%Y%m%d_%H%M").log

# You can delete the sdpa attn_implementation if you want to use flash attn
# --attn_implementation sdpa
