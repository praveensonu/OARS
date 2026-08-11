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
from gd_utils import GradDiffTrainer, GATrainer
from template import (
    LLAMA3_CHAT_TEMPLATE,
    qwen_chat_template,
    olmo_chat_template,
)


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

accelerator = Accelerator()


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unlearning on a fine-tuned model"
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
        help="Training stage: impair or warmup",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

class Config_gd:
    def __init__(self, model_name, dataset, stage):

        self.model_name = model_name
        self.dataset = dataset
        self.stage = stage

        self.loss_type = "ga"

        self.lr = 1e-4
        self.batch_size = 2
        self.gradient_accumulation_steps = 4

        self.weight_decay = 0.01
        self.max_length = 512

        self.selection = "no"

        # -------------------------------------------------
        # Impair / warmup configuration
        # -------------------------------------------------

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

        self.adaptor_path = (
            f"./outputs/{self.model_name}_{self.dataset}"
        )

        self.save_dir = (
            f"./outputs/unlearning/"
            f"{self.model_name}_{self.loss_type}_"
            f"{self.dataset}_{self.selection}_{self.stage}_model"
        )

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
# Chat formatting
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

    cfg = Config_gd(
        model_name=cli_args.model_name,
        dataset=cli_args.dataset,
        stage=cli_args.stage,
    )

    # -----------------------------------------------------
    # Print experiment configuration
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
        metrics.append(("Number of Steps", cfg.max_steps))

    elif cfg.stage == "warmup":
        metrics.append(("Number of Epochs", cfg.num_epochs))

    print("\n\n============ Check List ============\n")

    print(
        tabulate(
            metrics,
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Load forget data
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
    # Load retain data only if required
    # -----------------------------------------------------

    if cfg.loss_type != "ga":

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
        f"\nLoading the tokenizer: {cfg.model_id}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Load base model
    # -----------------------------------------------------

    print(
        f"\nLoading the model: {cfg.model_id}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # -----------------------------------------------------
    # Load fine-tuned LoRA adapter
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
    # Apply chat templates
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

    if cfg.loss_type != "ga":

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
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
    )

    if cfg.stage == "impair":
        training_kwargs["max_steps"] = cfg.max_steps
        training_kwargs["save_steps"] = cfg.save_steps

    elif cfg.stage == "warmup":
        training_kwargs["num_train_epochs"] = cfg.num_epochs
        training_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(
        **training_kwargs
)
    # -----------------------------------------------------
    # Dataset + trainer
    # -----------------------------------------------------

    if cfg.loss_type == "ga":

        dataset = SingleDataset(
            forget_data=forget,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
        )

        print(
            "\nLength of dataset:",
            len(dataset),
        )

        trainer = GATrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=custom_data_collator,
        )

    else:

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

        trainer = GradDiffTrainer(
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