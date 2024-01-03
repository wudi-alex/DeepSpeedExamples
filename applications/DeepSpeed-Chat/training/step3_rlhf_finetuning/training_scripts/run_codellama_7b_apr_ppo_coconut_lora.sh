#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=ksun
#SBATCH --job-name=ds_apr_ppo
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

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama2
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 ../main.py \
   --data_path /projects/ksun3/dwu25/apr_datasets_processing/coconut/data/apr_rlhf_coconut \
   --data_split 2,4,4 \
   --actor_model_name_or_path /projects/ksun3/dwu25/trained_models/ds_apr_sft \
   --critic_model_name_or_path /projects/ksun3/dwu25/trained_models/ds_apr_rm \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 100 \
   --max_prompt_seq_len 500 \
   --actor_learning_rate 5e-6 \
   --critic_learning_rate 2e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 2 \
   --critic_zero_stage 3 \
   --warmup_rate 0.05 \
   --actor_lora_dim 8 \
   --critic_lora_dim 8 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --enable_tensorboard \
   --tensorboard_path /projects/ksun3/dwu25/trained_models/ds_apr_ppo/tb \
   --enable_hybrid_engine \
   --dtype bf16 \
   --output_dir /projects/ksun3/dwu25/trained_models/ds_apr_ppo
