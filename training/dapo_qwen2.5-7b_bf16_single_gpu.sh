export TORCHDYNAMO_VERBOSE=1
#!/bin/bash
export WANDB_PROJECT=QeRL
export MASTER_ADDR=localhost
export MASTER_PORT=29523
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

MODEL_NAME="qwen2.5-7B-single-gpus"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

max_prompt_length=1024
max_completion_length=8192
max_seq_length=10000

perdevice_train_batch_size=2
gradient_accumulation_steps=1
lora_rank=16
lora_alpha=16


rl_mode="grpo"
beta=0.04
num_iterations=1
epsilon_high=0.2
loss_type="bnpo"  # bnpo equals to DAPO, the default is grpo (sample level and token level)
DATA_NAME="bigmath"  # dapomath, gsm8k, codeforces
optim="adamw_8bit"  # Optimizer type, adamw_8bit or adamw

if [ "$DATA_NAME" == "gsm8k" ]; then
    max_prompt_length=256
    max_completion_length=2048
    max_seq_length=2500
elif [ "$DATA_NAME" == "bigmath" ]; then
    max_prompt_length=1024
    max_completion_length=4096
    max_seq_length=5500
fi

if [ "$loss_type" == "bnpo" ]; then
    rl_mode="dapo"
    beta=0.0
    epsilon_high=0.28
    num_iterations=1
fi

lr=5e-6
ln=True

runname=${MODEL_NAME}_${lr}_${epsilon_high}_${DATA_NAME}_b${perdevice_train_batch_size}_g${gradient_accumulation_steps}_r${lora_rank}_${ln}

CKPTS_DIR=${CKPTS_DIR:-"./ckpt/${runname}"}

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=1 --main_process_port $MASTER_PORT \
    qerl.py  \
    --dataset $DATA_NAME \
    --model-name $MODEL_PATH \
    --output-dir $CKPTS_DIR \
    --use-vllm True \
    --learning-rate $lr \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type "cosine" \
    --optim $optim \
    --logging-steps 1 \
    --per-device-train-batch-size $perdevice_train_batch_size \
    --gradient-accumulation-steps $gradient_accumulation_steps \
    --num-generations $perdevice_train_batch_size \
    --max-prompt-length $max_prompt_length \
    --max-completion-length $max_completion_length \
    --num-train-epochs 1 \
    --save-steps 50 \
    --save-strategy "steps" \
    --save-total-limit 5 \
    --max-grad-norm 0.2 \
    --max-seq-length $max_seq_length \
    --lora-rank $lora_rank \
    --lora-alpha $lora_alpha \
    --fast-inference True \
    --vllm-gpu-memory-utilization 0.3 \
    --random-state 2025 \
    --loss-type $loss_type \
    --beta $beta \
    --epsilon-high $epsilon_high \
    --num-iterations $num_iterations \
    --mask-truncated-completions True \
    --run-name $runname \
    --ln $ln 