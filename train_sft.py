#!/usr/bin/env python3
"""
SFT (LoRA) Fine-Tuning Script for Qwen3.5-9B on NVIDIA A100-40GB
Optimized for ChatML / multi-turn dialogues with assistant-only loss masking.
"""

import os
import json
import argparse
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-9B with LoRA on A100")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking",
        help="Path to pre-trained base model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/sft/sft_train_final.jsonl",
        help="Path to training jsonl dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/sft/output_lora",
        help="Directory to save trained LoRA adapters and checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum token sequence length")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    return parser.parse_args()


def process_dialogue(messages: List[Dict[str, str]], tokenizer: Any, max_length: int) -> Dict[str, List[int]]:
    """
    Format multi-turn conversation into tokens and apply assistant-only loss masking.
    Mask all system prompts, user turns, and assistant headers with -100 in labels.
    """
    input_ids = []
    labels = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role in ["system", "user"]:
            prefix = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            p_ids = tokenizer.encode(prefix, add_special_tokens=False)
            input_ids.extend(p_ids)
            labels.extend([-100] * len(p_ids))
        elif role == "assistant":
            header = "<|im_start|>assistant\n"
            h_ids = tokenizer.encode(header, add_special_tokens=False)
            body = f"{content}<|im_end|>\n"
            b_ids = tokenizer.encode(body, add_special_tokens=False)
            input_ids.extend(h_ids + b_ids)
            labels.extend([-100] * len(h_ids) + b_ids)

    # Truncate if exceeds max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SFTDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    args = parse_args()
    print("=" * 60)
    print("Qwen3.5-9B SFT LoRA Training Setup")
    print(f"Base Model: {args.model_dir}")
    print(f"Dataset:    {args.data_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Batch Size: {args.batch_size} (accum={args.grad_accum}, effective={args.batch_size * args.grad_accum})")
    print(f"Epochs:     {args.epochs}, LR: {args.lr}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 60)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load and Prepare Dataset
    print("\n[1/4] Loading and preparing dataset...")
    raw_ds = load_dataset("json", data_files=args.data_path)["train"]
    split_ds = raw_ds.train_test_split(test_size=args.val_ratio, seed=42)
    train_raw = split_ds["train"]
    val_raw = split_ds["test"]
    print(f"Training samples: {len(train_raw)}, Validation samples: {len(val_raw)}")

    train_ds = train_raw.map(
        lambda ex: process_dialogue(ex["messages"], tokenizer, args.max_length),
        remove_columns=train_raw.column_names,
        desc="Formatting train data",
    )
    val_ds = val_raw.map(
        lambda ex: process_dialogue(ex["messages"], tokenizer, args.max_length),
        remove_columns=val_raw.column_names,
        desc="Formatting val data",
    )

    # 3. Load Model in BF16
    print("\n[2/4] Loading base model in BF16...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # 4. Inject LoRA Adapters
    print("\n[3/4] Initializing LoRA adapters...")
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "out_proj"
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        logging_first_step=True,
        dataloader_pin_memory=True,
    )

    data_collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # 6. Execute Training
    print("\n[4/4] Starting SFT training...")
    train_result = trainer.train()

    print("\nTraining completed! Evaluation results:")
    metrics = trainer.evaluate()
    print(metrics)

    # 7. Save Final Adapter
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"\nSaving final LoRA adapter to: {final_adapter_dir}")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print("LoRA Adapter successfully saved.")


if __name__ == "__main__":
    main()
