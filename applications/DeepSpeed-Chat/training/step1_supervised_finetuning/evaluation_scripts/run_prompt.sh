#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python3 prompt_eval.py \
    --model_name_or_path_baseline /work/cyh_deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/ \
    --model_name_or_path_finetune /work/cyh_deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/ \
    
