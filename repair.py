import argparse
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import PeftModel
from accelerate import Accelerator
from dotenv import load_dotenv
from tabulate import tabulate

from config import Config_repair_auto

from unlearning.template import (
    LLAMA3_CHAT_TEMPLATE,
    olmo_chat_template,
)

from unlearning.data_module import SingleDataset
from unlearning.collators import custom_data_collator


load_dotenv()

hf_token = os.getenv("HF_TOKEN")


# ---------------------------------------------------------
# Qwen template
# ---------------------------------------------------------

qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run repair experiments"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["muse", "bio"],
        help="Dataset",
    )

    parser.add_argument(
        "--selection",
        type=str,
        required=True,
        choices=[
            "emb",
            "random",
            "raslik",
            "oars",
        ],
        help="Repair sample selection method",
    )

    return parser.parse_args()


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
        return pd.read_json(
            path,
            lines=True,
        )

    raise ValueError(
        f"Unsupported file type: {path}"
    )


# ---------------------------------------------------------
# Repair one model/loss combination
# ---------------------------------------------------------

def run_repair(
    model_name,
    loss_type,
    dataset,
    selection,
):

    cfg = Config_repair_auto(
        model_name=model_name,
        loss_type=loss_type,
        dataset=dataset,
        selection=selection,
    )

    # -----------------------------------------------------
    # Checklist
    # -----------------------------------------------------

    metrics = [
        ("Model", cfg.model_name),
        ("Model ID", cfg.model_id),
        ("Dataset", cfg.dataset),
        ("Selection type", cfg.selection),
        ("Unlearning Loss", cfg.loss_type),
        ("Adaptor Path", cfg.adaptor_path),
        ("Retain Path", cfg.data_path),
        ("Save Path", cfg.save_dir),
        ("Number of Steps", cfg.max_steps),
    ]

    print(
        f"\n\n============ "
        f"Check List: "
        f"{model_name} | {loss_type} | "
        f"{dataset} | {selection} "
        f"============\n"
    )

    print(
        tabulate(
            metrics,
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Load repair dataset
    # -----------------------------------------------------

    data = read_file(
        cfg.data_path
    )

    print(
        f"\nRepair dataset shape: {data.shape}"
    )

    # -----------------------------------------------------
    # Apply chat template
    # -----------------------------------------------------

    if cfg.model_name == "llama":

        data["question"] = data["question"].apply(
            lambda x: LLAMA3_CHAT_TEMPLATE.format(
                question=x
            )
        )

    elif cfg.model_name == "qwen":

        data["question"] = data["question"].apply(
            lambda x: qwen_chat_template.format(
                question=x
            )
        )

    elif cfg.model_name == "olmo":

        data["question"] = data["question"].apply(
            lambda x: olmo_chat_template.format(
                question=x
            )
        )

    print(
        "\nExample repair question:\n"
    )

    print(
        data["question"].iloc[0]
    )

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

        tokenizer.pad_token = (
            "<|finetune_right_pad_id|>"
        )

    elif cfg.model_name in ["qwen", "olmo"]:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # -----------------------------------------------------
    # Base model
    # -----------------------------------------------------

    print(
        f"\nLoading base model: "
        f"{cfg.model_id}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=False,
    )

    model.config.use_cache = False

    # -----------------------------------------------------
    # Load unlearned LoRA adapter
    # -----------------------------------------------------

    print(
        f"\nLoading unlearned LoRA adapter "
        f"from:\n{cfg.adaptor_path}"
    )

    model = PeftModel.from_pretrained(
        model,
        cfg.adaptor_path,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset_obj = SingleDataset(
        data,
        tokenizer,
        max_length=cfg.max_length,
    )

    print(
        f"\nNumber of repair samples: "
        f"{len(dataset_obj)}"
    )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    training_args = TrainingArguments(
        per_device_train_batch_size=(
            cfg.batch_size
        ),

        learning_rate=cfg.lr,

        bf16=True,

        max_steps=cfg.max_steps,

        logging_dir=(
            f"{cfg.save_dir}/logs"
        ),

        eval_strategy="no",

        gradient_accumulation_steps=(
            cfg.gradient_accumulation_steps
        ),

        save_strategy="steps",

        save_steps=cfg.save_steps,

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
        train_dataset=dataset_obj,
        processing_class=tokenizer,
        data_collator=custom_data_collator,
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    trainer.train()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    model = model.cpu()

    model.save_pretrained(
        cfg.save_dir
    )

    tokenizer.save_pretrained(
        cfg.save_dir
    )

    print(
        f"\nModel and tokenizer saved to "
        f"{cfg.save_dir}"
    )

    # -----------------------------------------------------
    # Cleanup before next experiment
    # -----------------------------------------------------

    del trainer
    del model
    del tokenizer
    del dataset_obj
    del data
    del accelerator

    gc.collect()

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def experiment_available(cfg):
    """
    Check whether the required files for this repair experiment exist.

    Returns:
        available: bool
        reason: str
    """

    # Check unlearning adapter
    if not os.path.isdir(cfg.adaptor_path):
        return False, f"Missing adapter: {cfg.adaptor_path}"

    # Check repair dataset
    if not os.path.isfile(cfg.data_path):
        return False, f"Missing repair data: {cfg.data_path}"

    return True, "Ready"


def prepare_experiments(
    model_names,
    loss_types,
    dataset,
    selection,
):
    experiments = []

    for model_name in model_names:

        for loss_type in loss_types:

            cfg = Config_repair_auto(
                model_name=model_name,
                loss_type=loss_type,
                dataset=dataset,
                selection=selection,
            )

            available, reason = experiment_available(cfg)

            experiments.append(
                {
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "cfg": cfg,
                    "available": available,
                    "reason": reason,
                }
            )

    return experiments

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    cli_args = parse_args()

    model_names = [
        "llama",
        "olmo",
        "qwen",
    ]

    loss_types = [
        "ga",
        "npo",
        "snpo",
    ]

    # -----------------------------------------------------
    # Check all requested experiments first
    # -----------------------------------------------------

    experiments = prepare_experiments(
        model_names=model_names,
        loss_types=loss_types,
        dataset=cli_args.dataset,
        selection=cli_args.selection,
    )

    # -----------------------------------------------------
    # Show what is available / unavailable
    # -----------------------------------------------------

    overview = []

    for exp in experiments:

        cfg = exp["cfg"]

        status = (
            "RUN"
            if exp["available"]
            else "SKIP"
        )

        overview.append(
            [
                cfg.model_name,
                cfg.loss_type,
                cfg.dataset,
                cfg.selection,
                status,
                exp["reason"],
            ]
        )

    print(
        "\n\n"
        "============ Repair Plan ============\n"
    )

    print(
        tabulate(
            overview,
            headers=[
                "Model",
                "Loss",
                "Dataset",
                "Selection",
                "Status",
                "Reason",
            ],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Keep only valid experiments
    # -----------------------------------------------------

    runnable_experiments = [
        exp
        for exp in experiments
        if exp["available"]
    ]

    print(
        f"\nExperiments to run: "
        f"{len(runnable_experiments)}"
    )

    # -----------------------------------------------------
    # Nothing available
    # -----------------------------------------------------

    if len(runnable_experiments) == 0:

        print(
            "\nNo repair experiments are available."
        )

        print(
            "Nothing will be run."
        )

    # -----------------------------------------------------
    # Run available experiments
    # -----------------------------------------------------

    else:

        print(
            "\nThe following experiments "
            "will be repaired:\n"
        )

        for exp in runnable_experiments:

            cfg = exp["cfg"]

            print(
                f"  {cfg.model_name} "
                f"+ {cfg.loss_type} "
                f"+ {cfg.dataset} "
                f"+ {cfg.selection}"
            )

        print(
            "\n====================================\n"
        )

        for exp in runnable_experiments:

            cfg = exp["cfg"]

            run_repair(
                model_name=cfg.model_name,
                loss_type=cfg.loss_type,
                dataset=cfg.dataset,
                selection=cfg.selection,
            )