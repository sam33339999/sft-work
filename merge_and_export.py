#!/usr/bin/env python3
"""
Merge trained LoRA weights back into the base Qwen3.5-9B model
for direct deployment via vLLM or Hugging Face.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into Qwen3.5-9B")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking",
        help="Path to base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="/content/sft/output_lora/final_adapter",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/sft/Qwen3.5-9B-FineTuned",
        help="Directory to save merged full model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Merging LoRA Adapter into Base Model")
    print(f"Base Model: {args.base_model_dir}")
    print(f"Adapter:    {args.adapter_dir}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)

    print("\n[1/3] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    print("\n[2/3] Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged_model = model.merge_and_unload()

    print(f"\n[3/3] Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, max_shard_size="5GB", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Copy ancillary config files if needed
    for fname in ["chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]:
        src = os.path.join(args.base_model_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    print(f"\nSuccessfully merged and exported to: {args.output_dir}")


if __name__ == "__main__":
    main()
