import argparse
import os

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from dotenv import load_dotenv

from config import Config_ft
from unlearning.template import LLAMA3_CHAT_TEMPLATE
from unlearning.data_module import SingleDataset
from unlearning.collators import custom_data_collator


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM using LoRA"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["llama", "olmo", "qwen"],
        help="Model to fine-tune",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["muse", "bio"],
        help="Dataset to use",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def read_file(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)

    elif path.endswith(".json"):
        return pd.read_json(path)

    elif path.endswith(".parquet"):
        return pd.read_parquet(path)

    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)

    else:
        raise ValueError(f"Unsupported file format: {path}")


# ---------------------------------------------------------
# Chat templates
# ---------------------------------------------------------

qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


olmo_chat_template = """<|endoftext|><|user|>{question}<|assistant|>"""


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # Parse terminal arguments
    cli_args = parse_args()

    # Load configuration
    cfg = Config_ft(
        model_name=cli_args.model_name,
        dataset=cli_args.dataset,
    )

    print("=" * 60)
    print(f"Model      : {cfg.model_name}")
    print(f"Model ID   : {cfg.model_id}")
    print(f"Dataset    : {cfg.dataset}")
    print(f"Data path  : {cfg.data_path}")
    print(f"Save dir   : {cfg.save_dir}")
    print("=" * 60)

    # HuggingFace token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    data = read_file(cfg.data_path)

    print(f"Dataset shape: {data.shape}")

    # -----------------------------------------------------
    # Apply model-specific chat template
    # -----------------------------------------------------

    if cfg.model_name == "qwen":

        data["question"] = data["question"].apply(
            lambda x: qwen_chat_template.format(question=x)
        )

    elif cfg.model_name == "llama":

        data["question"] = data["question"].apply(
            lambda x: LLAMA3_CHAT_TEMPLATE.format(question=x)
        )

    elif cfg.model_name == "olmo":

        data["question"] = data["question"].apply(
            lambda x: olmo_chat_template.format(question=x)
        )

    print("\nExample formatted question:\n")
    print(data["question"].iloc[0])

    # -----------------------------------------------------
    # Accelerator
    # -----------------------------------------------------

    accelerator = Accelerator()

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        token=hf_token,
    )

    if cfg.model_name == "llama":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    elif cfg.model_name in ["qwen", "olmo"]:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=False,
    )

    model.config.use_cache = False

    # -----------------------------------------------------
    # LoRA
    # -----------------------------------------------------

    lora_config = LoraConfig(
        r=cfg.LoRA_r,
        lora_alpha=cfg.LoRA_alpha,
        lora_dropout=cfg.LoRA_dropout,
        target_modules=cfg.LoRa_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = SingleDataset(
        data,
        tokenizer,
        max_length=cfg.max_length,
    )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        bf16=True,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,

        logging_dir=f"{cfg.save_dir}/logs",

        eval_strategy="no",

        gradient_accumulation_steps=(
            cfg.gradient_accumulation_steps
        ),

        save_strategy="epoch",
        save_total_limit=10,

        output_dir=cfg.save_dir,

        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=custom_data_collator,
    )

    trainer.train()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    model = model.cpu()

    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    print(f"\nModel and tokenizer saved to {cfg.save_dir}")

    # -----------------------------------------------------
    # Push to Hugging Face Hub
    # -----------------------------------------------------

    repo_name = f"rep-shift/{cfg.model_name}_{cfg.dataset}"

    try:
        model.push_to_hub(
            repo_name,
            token=hf_token,
        )

        tokenizer.push_to_hub(
            repo_name,
            token=hf_token,
        )

        print(f"Pushed model to HuggingFace: {repo_name}")

    except Exception as e:
        print(f"Could not push to hub: {e}")


if __name__ == "__main__":
    main()