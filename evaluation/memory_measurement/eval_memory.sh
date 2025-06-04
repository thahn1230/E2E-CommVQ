#!/bin/bash


START=8192
END=131072
STEP=8192

MODEL_NAME="../training/output/llama7b_010801_int1_pack/checkpoint-40000"

CUDA_VISIBLE_DEVICES=0 python -u eval_memory.py --s_len $START --e_len $END \
    --model_name ${MODEL_NAME} \
    --attn_implementation flash_attention_2 \
    --step $STEP \
    --model_version ${MODEL_NAME}_${START}_${END}_${STEP}
