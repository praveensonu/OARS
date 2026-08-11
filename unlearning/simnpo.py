import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from accelerate import Accelerator
from tabulate import tabulate
from dotenv import load_dotenv

from data_module import DualDatasetRandom, SingleDataset
from collators import custom_gd_collator_forget, custom_data_collator
from snpo_utils import SNPOR, SNPO
from template import (
    LLAMA3_CHAT_TEMPLATE,
    qwen_chat_template,
    olmo_chat_template,
)


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

accelerator = Accelerator()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SimNPO unlearning"
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


class Config_snpo:
    def __init__(self, model_name, dataset, stage):

        self.model_name = model_name
        self.dataset = dataset
        self.stage = stage

        self.loss_type = "snpo"

        self.lr = 1e-4
        self.batch_size = 2
        self.gradient_accumulation_steps = 4

        self.weight_decay = 0.01
        self.max_length = 512

        # Fixed selection
        self.selection = "no"

        # SimNPO parameters
        self.delta = 0.0
        self.beta = 3.5

        # Stage-specific configuration
        if self.stage == "impair":
            self.max_steps = 100
            self.num_epochs = None

        elif self.stage == "warmup":
            self.max_steps = None
            self.num_epochs = 1

        self.save_steps = 20

        model_ids = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "olmo": "allenai/OLMo-2-0425-1B-Instruct",
            "qwen": "Qwen/Qwen2.5-3B-Instruct",
        }

        self.model_id = model_ids[self.model_name]

        # Finetuned adapter
        self.adaptor_path = (
            f"./outputs/{self.model_name}_{self.dataset}"
        )

        # Output path
        self.save_dir = (
            f"./outputs/unlearning/"
            f"{self.model_name}_{self.loss_type}_"
            f"{self.dataset}_{self.selection}_{self.stage}_model"
        )

        # Dataset paths
        self.forget_path = (
            f"../data/{self.dataset}_forget.parquet"
        )

        self.retain_path = (
            f"../data/{self.dataset}_retain.parquet"
        )


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


def main():

    cli_args = parse_args()

    cfg = Config_snpo(
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
        ("Forget Path", cfg.forget_path),
        ("Adapter Path", cfg.adaptor_path),
        ("Save Path", cfg.save_dir),
        ("SimNPO Delta", cfg.delta),
        ("SimNPO Beta", cfg.beta),
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
    # Load retain set only if needed
    # -----------------------------------------------------

    if cfg.loss_type != "snpo":

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
    # Base model
    # -----------------------------------------------------

    print(
        f"\nLoading model: {cfg.model_id}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # -----------------------------------------------------
    # Load finetuned LoRA adapter
    # -----------------------------------------------------

    print(
        f"\nLoading LoRA adapter from "
        f"{cfg.adaptor_path}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        cfg.adaptor_path,
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    # -----------------------------------------------------
    # Apply chat template
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

    if cfg.loss_type != "snpo":

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
    )

    # Impair -> train using steps
    if cfg.stage == "impair":

        training_kwargs["max_steps"] = (
            cfg.max_steps
        )

        training_kwargs["save_strategy"] = (
            "steps"
        )

        training_kwargs["save_steps"] = (
            cfg.save_steps
        )

    # Warmup -> train for exactly 1 epoch
    elif cfg.stage == "warmup":

        training_kwargs["num_train_epochs"] = (
            cfg.num_epochs
        )

        training_kwargs["save_strategy"] = (
            "epoch"
        )

    training_args = TrainingArguments(
        **training_kwargs
    )

    # -----------------------------------------------------
    # Dataset + trainer
    # -----------------------------------------------------

    if cfg.loss_type == "snpo":

        print(
            "\nNow running SimNPO without Retain"
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

        trainer = SNPO(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=custom_data_collator,
        )

    else:

        print(
            "\nNow running SimNPO with Retain"
        )

        dataset = DualDatasetRandom(
            forget_data=forget,
            retain_data=retain,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
        )

        print(
            "\nLength of dataset:",
            len(dataset),
        )

        trainer = SNPOR(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=custom_gd_collator_forget,
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    trainer.train()

    accelerator.wait_for_everyone()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    model.save_pretrained(
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