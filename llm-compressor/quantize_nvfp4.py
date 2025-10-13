from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier,GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from datasets import load_dataset
import argparse



def quantize_model(model_id, skip_generation, push_to_hub):
    
    MODEL_ID = model_id
    if 'qwen' in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    elif 'llama' in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="bfloat16", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp4 with per group 16 via ptq
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    if not skip_generation:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=100)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("==========================================\n\n")

    # Save to disk in compressed-tensors format.
    SAVE_DIR = "./qmodel/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16-GPTQ"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    # if push_to_hub:
    #     model.push_to_hub(SAVE_DIR)
    #     tokenizer.push_to_hub(SAVE_DIR)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    quantize_model(args.model, args.skip_generation, args.push_to_hub)
