#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline codellama/CodeLlama-7b-hf \
    --model_name_or_path_finetune /projects/ksun3/dwu25/trained_models/ds_apr_sft
