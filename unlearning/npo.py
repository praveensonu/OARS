import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from accelerate import Accelerator
from peft import PeftModel
from tabulate import tabulate
from dotenv import load_dotenv

from npo_utils import *
from data_module import *
from template import (
    LLAMA3_CHAT_TEMPLATE,
    qwen_chat_template,
    olmo_chat_template,
)
from collators import (
    custom_gd_collator_forget,
    custom_data_collator,
)


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

accelerator = Accelerator()


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NPO unlearning"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["llama", "olmo", "qwen"],
        help="Model name",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["muse", "bio"],
        help="Dataset name",
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["impair", "warmup"],
        help="Training stage",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

class Config_npo:
    def __init__(self, model_name, dataset, stage):

        self.model_name = model_name
        self.dataset = dataset
        self.stage = stage

        self.loss_type = "npo"

        self.lr = 1e-4
        self.batch_size = 1
        self.gradient_accumulation_steps = 8

        self.weight_decay = 0.01
        self.max_length = 512

        self.selection = "no"

        # NPO parameters
        self.npo_beta = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0

        # -------------------------------------------------
        # Stage-specific training
        # -------------------------------------------------

        if self.stage == "impair":
            self.max_steps = 100
            self.num_epochs = None

        elif self.stage == "warmup":
            self.max_steps = None
            self.num_epochs = 1

        self.save_steps = 20

        # -------------------------------------------------
        # Model IDs
        # -------------------------------------------------

        model_ids = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "olmo": "allenai/OLMo-2-0425-1B-Instruct",
            "qwen": "Qwen/Qwen2.5-3B-Instruct",
        }

        self.model_id = model_ids[self.model_name]

        # -------------------------------------------------
        # Fine-tuned adapter
        # -------------------------------------------------

        self.adaptor_path = (
            f"./outputs/{self.model_name}_{self.dataset}"
        )

        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        self.save_dir = (
            f"./outputs/unlearning/"
            f"{self.model_name}_{self.loss_type}_"
            f"{self.dataset}_{self.selection}_{self.stage}_model"
        )

        # -------------------------------------------------
        # Dataset paths
        # -------------------------------------------------

        self.forget_path = (
            f"../data/{self.dataset}_forget.parquet"
        )

        self.retain_path = (
            f"../data/{self.dataset}_retain.parquet"
        )


# ---------------------------------------------------------
# File reader
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
        raise ValueError(
            f"Unsupported file format: {path}"
        )


# ---------------------------------------------------------
# Template formatting
# ---------------------------------------------------------

def make_template_format(df, model_name):

    if model_name == "llama":
        df["question"] = df["question"].apply(
            lambda x: LLAMA3_CHAT_TEMPLATE.format(
                question=x
            )
        )

    elif model_name == "qwen":
        df["question"] = df["question"].apply(
            lambda x: qwen_chat_template.format(
                question=x
            )
        )

    elif model_name == "olmo":
        df["question"] = df["question"].apply(
            lambda x: olmo_chat_template.format(
                question=x
            )
        )

    return df


def normalize_columns(df):

    return df.rename(
        columns={
            "prompt": "question",
            "generation": "answer",
        }
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    cli_args = parse_args()

    cfg = Config_npo(
        model_name=cli_args.model_name,
        dataset=cli_args.dataset,
        stage=cli_args.stage,
    )

    # -----------------------------------------------------
    # Checklist
    # -----------------------------------------------------

    metrics = [
        ("Model", cfg.model_id),
        ("Model name", cfg.model_name),
        ("Dataset", cfg.dataset),
        ("Stage", cfg.stage),
        ("Selection type", cfg.selection),
        ("Unlearning Loss", cfg.loss_type),
        ("Adapter Path", cfg.adaptor_path),
        ("Forget Path", cfg.forget_path),
        ("Save Path", cfg.save_dir),
    ]

    if cfg.stage == "impair":
        metrics.append(
            ("Number of Steps", cfg.max_steps)
        )

    elif cfg.stage == "warmup":
        metrics.append(
            ("Number of Epochs", cfg.num_epochs)
        )

    print("\n\n============ Check List ============\n")

    print(
        tabulate(
            metrics,
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Load forget set
    # -----------------------------------------------------

    print("\nLoading the forget set")

    forget = read_file(
        cfg.forget_path
    )

    forget = normalize_columns(forget)

    print(
        f"There are {len(forget)} samples "
        f"in the forget set"
    )

    # -----------------------------------------------------
    # Load retain if necessary
    # -----------------------------------------------------

    if cfg.loss_type != "npo":

        print("\nLoading the retain set")

        retain = read_file(
            cfg.retain_path
        )

        retain = normalize_columns(retain)

        print(
            f"There are {len(retain)} samples "
            f"in the retain set"
        )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    print(
        f"\nLoading tokenizer: {cfg.model_id}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        token=hf_token,
    )

    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Policy/base model
    # -----------------------------------------------------

    print(
        f"\nLoading base model: {cfg.model_id}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    print("Base model loaded.")

    # -----------------------------------------------------
    # Load LoRA adapter on policy model
    # -----------------------------------------------------

    print(
        f"\nLoading LoRA adapter from "
        f"{cfg.adaptor_path}"
    )

    policy_model = PeftModel.from_pretrained(
        base_model,
        cfg.adaptor_path,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )

    policy_model.config.use_cache = False

    policy_model.print_trainable_parameters()

    # -----------------------------------------------------
    # Reference model
    # -----------------------------------------------------

    print(
        f"\nLoading reference model: {cfg.model_id}"
    )

    ref_model_base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    ref_model = PeftModel.from_pretrained(
        ref_model_base,
        cfg.adaptor_path,
        torch_dtype=torch.bfloat16,
    )

    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

    # -----------------------------------------------------
    # Apply templates
    # -----------------------------------------------------

    forget = make_template_format(
        forget,
        cfg.model_name,
    )

    print(
        "\nForget example:\n",
        forget["question"].iloc[0],
        "\nAnswer:\n",
        forget["answer"].iloc[0],
    )

    if cfg.loss_type != "npo":

        retain = make_template_format(
            retain,
            cfg.model_name,
        )

        print(
            "\nRetain example:\n",
            retain["question"].iloc[0],
            "\nAnswer:\n",
            retain["answer"].iloc[0],
        )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    training_kwargs = dict(
        output_dir=cfg.save_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        logging_dir=f"{cfg.save_dir}/logs",
        eval_strategy="no",
        label_names=["labels"],
        bf16=True,
        gradient_accumulation_steps=(
            cfg.gradient_accumulation_steps
        ),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Impair = step-based
    if cfg.stage == "impair":

        training_kwargs["max_steps"] = cfg.max_steps
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["save_steps"] = cfg.save_steps

    # Warmup = exactly one epoch
    elif cfg.stage == "warmup":

        training_kwargs["num_train_epochs"] = (
            cfg.num_epochs
        )

        training_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(
        **training_kwargs
    )

    # -----------------------------------------------------
    # Dataset + trainer
    # -----------------------------------------------------

    if cfg.loss_type == "npo":

        print(
            "\nNow running NPO without Retain"
        )

        dataset = SingleDataset(
            forget_data=forget,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
        )

        print(
            "\nLength of dataset:",
            len(dataset),
        )

        trainer = NPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            beta=cfg.npo_beta,
            data_collator=custom_data_collator,
            gamma=cfg.npo_forget_gamma,
            alpha=cfg.npo_retain_alpha,
        )

    else:

        print(
            "\nNow running NPO with Retain"
        )

        dataset = DualDatasetRandom(
            forget_data=forget,
            retain_data=retain,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            question_key="question",
            answer_key="answer",
        )

        print(
            "\nLength of dataset:",
            len(dataset),
        )

        trainer = RetainNPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            beta=cfg.npo_beta,
            data_collator=custom_gd_collator_forget,
            gamma=cfg.npo_forget_gamma,
            alpha=cfg.npo_retain_alpha,
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    trainer.train()

    accelerator.wait_for_everyone()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    policy_model.save_pretrained(
        cfg.save_dir
    )

    tokenizer.save_pretrained(
        cfg.save_dir
    )

    print(
        f"\nForget LoRA adapter saved at "
        f"{cfg.save_dir}"
    )


if __name__ == "__main__":
    main()