# NVFP4 Quantization with llm-compressor

---

## Installation

Because the versions of `compressed_tensors` required for quantization and inference differ, it’s recommended to use two separate Conda environments:

1) Quantization environment: `llmcompressor` (for NVFP4 quantization)  
2) Inference environment: `vllm` (for benchmarking with vLLM)

---

## 1. Quantize to NVFP4

```bash
# 1) Create and activate the quantization environment
conda create -n llmcompressor python=3.12 -y
conda activate llmcompressor

# 2) Install dependencies
pip install -e .
pip install nvidia-ml-py

# 3) Quantize model to NVFP4 (example commands)
python quantize_nvfp4.py --model Qwen/Qwen2.5-3B
python quantize_nvfp4.py --model Qwen/Qwen2.5-7B
python quantize_nvfp4.py --model Qwen/Qwen2.5-14B
python quantize_nvfp4.py --model Qwen/Qwen2.5-32B
python quantize_nvfp4.py --model meta-llama/Llama-3.1-8B
```

After quantization, note the NVFP4 weight directory for each model, e.g., Qwen2.5-3B-NVFP4A16 (used later as the input path to --model for inference).

## Offline Inference Benchmarking Setup

- Precision: NVFP4
- Models:
  - Qwen2.5-3B / 7B / 14B / 32B (priority)
  - LLaMA-3.1-8B
- Model format: MXFP4 Linear + LoRA  
  - LoRA rank: default 32 (can later ablate with 16 / 64)
- I/O:
  - input length = 1024
  - output tokens = 4096–8192
  - batch size = 1
- Dataset: openai/gsm8k (for speed-only tests, random data is also acceptable)
- Inference frameworks: Transformers, vLLM, SGLang (examples below use vLLM)
- Phase: simulating RL Rollout

---

## 2. Inference and Benchmarking with vLLM

```bash
# 1) Create and activate the inference environment
conda create -n vllm python=3.12 -y
conda activate vllm

# 2) Install dependencies
pip install vllm
pip install peft
pip install pandas
```

Script description  
test_scripts/efficiency_test.py: measures prefill and decode throughput separately, covering long outputs (4096–8192) and RL Rollout scenarios with batch_size=1 and batch_size>1.

Example (Qwen/Qwen2.5-3B):

1. Full precision (baseline)
```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model Qwen/Qwen2.5-3B \
  --batch_size 1
```

2. NVFP4-quantized base model
```bash
# Qwen2.5-3B-NVFP4A16 is the local path to the quantized artifact (example name)
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-3B-NVFP4A16 \
  --batch_size 1
```

3. NVFP4 + LoRA (bf16)
```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-3B-NVFP4A16 \
  --add_lora \
  --lora_path bunnycore/qwen-2.5-3b-lora_model \
  --batch_size 1
```

---

## 3. Generate Dummy LoRA (for pure speed baselines)

If you only care about speed and not accuracy, you can use a zero-initialized Dummy LoRA to simulate the overhead of loading the LoRA branch. (Benchmark within the vLLM environment as well.)

Example: Generate Dummy LoRA (rank=64) for Qwen/Qwen2.5-7B
```bash
python test_scripts/generate_dummy_lora.py \
  --base_model Qwen/Qwen2.5-7B \
  --lora_rank 64 \
  --lora_path /path/to/qwen25_7b_lora_init_default
```

Then replace --lora_path in efficiency_test.py with the Dummy LoRA path:

```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-7B-NVFP4A16 \
  --add_lora \
  --lora_path /path/to/qwen25_7b_lora_init_default \
  --batch_size 1
```

Note: vLLM preallocates memory according to max_lora_rank for optimal performance. In efficiency_test.py, the default max_lora_rank is 16. If you change the LoRA rank for ablation, ensure max_lora_rank in efficiency_test.py is greater than the LoRA rank you pass in.