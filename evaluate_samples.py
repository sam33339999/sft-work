#!/usr/bin/env python3
"""
Test and compare generation capability before and after SFT.
"""

import argparse
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-9B")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="/content/sft/output_lora/final_adapter",
    )
    return parser.parse_args()


TEST_PROMPTS = [
    {
        "system": "你是一位熟悉台灣室內設計與建築工地的助理，精通工地台語、日語外來語與國語工程用語的對照，回答以台灣繁體中文為主，精準、實用。",
        "user": "我家廚房跟廁所門對門，這樣有什麼問題嗎？如果不改門有什麼折衷做法？",
    },
    {
        "system": "你是一位熟悉台灣工地用語與裝修術語的助理，用台灣繁體中文回答，說明簡短清楚，並會補充同一個詞的其他講法。",
        "user": "請問什麼是「放樣」？師傅說的「墨斗」是做什麼用的？",
    },
    {
        "system": "你是一位資深的台灣室內設計師，正在跟屋主溝通，善於用白話解釋裝修術語與工序，語氣親切專業。",
        "user": "為什麼裝修一定要先做拆除跟保護？不能木作先進場嗎？",
    },
]


def main():
    args = parse_args()
    print("=" * 60)
    print("Evaluating Fine-Tuned Model Capability")
    print(f"Base Model: {args.model_dir}")
    print(f"Adapter:    {args.adapter_dir}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    import os
    if os.path.exists(args.adapter_dir):
        print(f"Loading LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        print("Adapter not found, testing base model...")
        model = base_model

    model.eval()

    for idx, p in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{idx}] 測試問題: {p['user']}")
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("-" * 50)
        print("回答內容:")
        print(resp.strip())
        print("-" * 50)


if __name__ == "__main__":
    main()
