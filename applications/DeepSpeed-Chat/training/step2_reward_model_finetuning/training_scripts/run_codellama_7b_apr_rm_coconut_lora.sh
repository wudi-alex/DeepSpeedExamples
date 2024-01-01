#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=ksun
#SBATCH --job-name=ds_apr_rm
#SBATCH --output=/projects/ksun3/%u/sbatch_log/%x-%N-%j.out
#SBATCH --error=/projects/ksun3/%u/sbatch_log/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=480G
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00

set echo
umask 0022

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10

source 	~/Anaconda/etc/profile.d/conda.sh
conda activate dschat

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/projects/ksun3/dwu25/trained_models/ds_apr_rm
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

deepspeed ../main.py \
   --data_path /projects/ksun3/dwu25/apr_datasets_processing/coconut/data/apr_rlhf_coconut \
   --data_split 2,4,4 \
   --model_name_or_path codellama/CodeLlama-7b-hf \
   --per_device_train_batch_size 64 \
   --per_device_eval_batch_size 128 \
   --max_seq_len 600 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --output_dir /projects/ksun3/dwu25/trained_models/ds_apr_rm \
   --eval_interval 1000
