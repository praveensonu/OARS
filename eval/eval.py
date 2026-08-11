import argparse
import json
import os
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch

from dotenv import load_dotenv
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as hf_logging,
)

from config import Config_eval

from eval_utils import (
    compute_fq_scores,
    compute_mu_scores,
    read_file,
)

from unlearning.template import (
    LLAMA3_CHAT_TEMPLATE,
    qwen_chat_template,
    olmo_chat_template,
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------

LOSS_TYPES = [
    "ga",
    "npo",
    "snpo",
]

SELECTIONS = [
    "emb",
    "random",
    "raslik",
    "oars",
]


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Automatically evaluate pre, impair, and repair models"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "llama",
            "olmo",
            "qwen",
        ],
        help="Model to evaluate",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "muse",
            "bio",
        ],
        help="Dataset",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------

def make_template_format(
    df,
    model_name,
):

    df = df.copy()

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


# ---------------------------------------------------------
# Build all possible experiments
# ---------------------------------------------------------

def build_experiments(
    model_name,
    dataset,
):

    experiments = []

    # =====================================================
    # 1. Pre-unlearning
    # =====================================================

    experiments.append(
        Config_eval(
            model_name=model_name,
            dataset=dataset,
            experiment_type="pre",
        )
    )

    # =====================================================
    # 2. Impair
    # =====================================================

    for loss_type in LOSS_TYPES:

        experiments.append(
            Config_eval(
                model_name=model_name,
                dataset=dataset,
                experiment_type="impair",
                loss_type=loss_type,
            )
        )

    # =====================================================
    # 3. Repair
    # =====================================================

    for loss_type in LOSS_TYPES:

        for selection in SELECTIONS:

            experiments.append(
                Config_eval(
                    model_name=model_name,
                    dataset=dataset,
                    experiment_type="repair",
                    loss_type=loss_type,
                    selection=selection,
                )
            )

    return experiments


# ---------------------------------------------------------
# Check experiment availability
# ---------------------------------------------------------

def check_experiment(cfg):

    if not os.path.isdir(cfg.adaptor_path):

        return (
            False,
            f"Adapter not found: {cfg.adaptor_path}",
        )

    return True, "Ready"


# ---------------------------------------------------------
# Show evaluation plan
# ---------------------------------------------------------

def prepare_evaluation(
    experiments,
):

    table = []

    runnable = []

    for cfg in experiments:

        available, reason = (
            check_experiment(cfg)
        )

        if available:
            status = "RUN"
            runnable.append(cfg)

        else:
            status = "SKIP"

        table.append(
            [
                cfg.experiment_type,
                cfg.model_name,
                cfg.loss_type or "-",
                cfg.selection or "-",
                status,
                cfg.adaptor_path,
            ]
        )

    print(
        "\n\n"
        "============ Evaluation Plan ============\n"
    )

    print(
        tabulate(
            table,
            headers=[
                "Type",
                "Model",
                "Loss",
                "Selection",
                "Status",
                "Adapter",
            ],
            tablefmt="github",
        )
    )

    print(
        f"\nExperiments available: "
        f"{len(runnable)} / {len(experiments)}"
    )

    if runnable:

        print(
            "\nThe following experiments "
            "will be evaluated:\n"
        )

        for cfg in runnable:

            print(
                f"  - {cfg.experiment_name}"
            )

    return runnable


# ---------------------------------------------------------
# Generate unique PEFT adapter name
# ---------------------------------------------------------

def get_adapter_name(cfg):

    if cfg.experiment_type == "pre":

        return "adapter_pre"

    elif cfg.experiment_type == "impair":

        return (
            f"adapter_impair_"
            f"{cfg.loss_type}"
        )

    elif cfg.experiment_type == "repair":

        return (
            f"adapter_repair_"
            f"{cfg.loss_type}_"
            f"{cfg.selection}"
        )


# ---------------------------------------------------------
# Evaluate one adapter
# ---------------------------------------------------------

def run_one(
    cfg,
    model,
    tokenizer,
    embedding_model,
    forget_data,
    test_data,
    audit_data,
    all_results,
    results_json_path,
    out_dir,
):

    print(
        "\n\n"
        "========================================"
    )

    print(
        f"Evaluating: {cfg.experiment_name}"
    )

    print(
        "========================================\n"
    )

    print(
        tabulate(
            [
                (
                    "Experiment",
                    cfg.experiment_type,
                ),
                (
                    "Model",
                    cfg.model_name,
                ),
                (
                    "Dataset",
                    cfg.dataset,
                ),
                (
                    "Loss",
                    cfg.loss_type or "N/A",
                ),
                (
                    "Selection",
                    cfg.selection or "N/A",
                ),
                (
                    "Adapter",
                    cfg.adaptor_path,
                ),
            ],
            headers=[
                "Field",
                "Value",
            ],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Load/select adapter
    # -----------------------------------------------------

    adapter_name = get_adapter_name(cfg)

    if adapter_name not in model.peft_config:

        print(
            f"\nLoading adapter: "
            f"{cfg.adaptor_path}"
        )

        model.load_adapter(
            cfg.adaptor_path,
            adapter_name=adapter_name,
        )

    model.set_adapter(
        adapter_name
    )

    # -----------------------------------------------------
    # Forget efficacy
    # -----------------------------------------------------

    print(
        "\nCalculating forget efficacy..."
    )

    (
        forget,
        all_scores_fe,
        fq,
        f_ppl,
    ) = compute_fq_scores(
        df=forget_data,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # -----------------------------------------------------
    # Test utility
    # -----------------------------------------------------

    print(
        "\nCalculating Test utility..."
    )

    (
        test,
        all_scores_mu,
        mu,
        test_ppl,
    ) = compute_mu_scores(
        df=test_data,
        model=model,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        device=device,
    )

    # -----------------------------------------------------
    # Audit utility
    #
    # IMPORTANT:
    # AU is now calculated exactly like Test.
    # -----------------------------------------------------

    print(
        "\nCalculating AU..."
    )

    (
        audit,
        all_scores_au,
        au,
        audit_ppl,
    ) = compute_mu_scores(
        df=audit_data,
        model=model,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        device=device,
    )

    # -----------------------------------------------------
    # Save detailed result DataFrames
    # -----------------------------------------------------

    suffix = (
        cfg.experiment_name
    )

    forget.to_parquet(
        out_dir
        / f"forget_{suffix}.parquet",
        index=False,
    )

    test.to_parquet(
        out_dir
        / f"test_{suffix}.parquet",
        index=False,
    )

    audit.to_parquet(
        out_dir
        / f"audit_{suffix}.parquet",
        index=False,
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    metrics = [
        (
            "Experiment",
            cfg.experiment_name,
        ),
        (
            "FQ",
            fq.item(),
        ),
        (
            "Test",
            mu.item(),
        ),
        (
            "AU",
            au.item(),
        ),
        (
            "PPL-F",
            f_ppl.item(),
        ),
        (
            "PPL-T",
            test_ppl.item(),
        ),
        (
            "PPL-A",
            audit_ppl.item(),
        ),
    ]

    print(
        "\n\n"
        "============ RESULTS ============\n"
    )

    print(
        tabulate(
            metrics,
            headers=[
                "Metric",
                "Value",
            ],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------
    # Store JSON result
    # -----------------------------------------------------

    key = (
        cfg.experiment_name
    )

    all_results[key] = {

        "experiment_type": (
            cfg.experiment_type
        ),

        "model": (
            cfg.model_name
        ),

        "dataset": (
            cfg.dataset
        ),

        "loss_type": (
            cfg.loss_type
        ),

        "selection": (
            cfg.selection
        ),

        "adapter_path": (
            cfg.adaptor_path
        ),

        "FQ": (
            fq.item()
        ),

        "Test Utility": (
            mu.item()
        ),

        "AU": (
            au.item()
        ),

        "forget_scores": (
            all_scores_fe.tolist()
        ),

        "test_scores": (
            all_scores_mu.tolist()
        ),

        "audit_scores": (
            all_scores_au.tolist()
        ),

        "PPL-F": (
            f_ppl.item()
        ),

        "PPL-T": (
            test_ppl.item()
        ),

        "PPL-A": (
            audit_ppl.item()
        ),
    }

    with open(
        results_json_path,
        "w",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to: "
        f"{results_json_path}"
    )

    return all_results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    cli_args = parse_args()

    MODEL_NAME = (
        cli_args.model_name
    )

    DATASET = (
        cli_args.dataset
    )

    # -----------------------------------------------------
    # Result directories
    # -----------------------------------------------------

    out_dir = Path(
        "../results/evaluation"
    )

    score_dir = Path(
        "../results/scores"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_json_path = (
        score_dir
        / f"{MODEL_NAME}_{DATASET}.json"
    )

    # -----------------------------------------------------
    # Discover available experiments
    # -----------------------------------------------------

    experiments = build_experiments(
        model_name=MODEL_NAME,
        dataset=DATASET,
    )

    runnable = prepare_evaluation(
        experiments
    )

    # -----------------------------------------------------
    # Nothing available
    # -----------------------------------------------------

    if not runnable:

        print(
            "\nNo available adapters found."
        )

        print(
            "Nothing to evaluate."
        )

        return

    # -----------------------------------------------------
    # Base config
    # -----------------------------------------------------

    cfg0 = runnable[0]

    # -----------------------------------------------------
    # Verify evaluation datasets
    # -----------------------------------------------------

    required_data = [
        cfg0.forget_path,
        cfg0.test_path,
        cfg0.audit_path,
    ]

    for path in required_data:

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Evaluation dataset "
                f"not found: {path}"
            )

    # -----------------------------------------------------
    # Existing results
    # -----------------------------------------------------

    if results_json_path.exists():

        with open(
            results_json_path,
            "r",
        ) as f:

            all_results = (
                json.load(f)
            )

    else:

        all_results = {}

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    print(
        f"\nLoading tokenizer: "
        f"{cfg0.model_id}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            cfg0.model_id,
            token=hf_token,
        )
    )

    tokenizer.pad_token = tokenizer.eos_token
        

    # -----------------------------------------------------
    # Evaluation datasets
    # -----------------------------------------------------

    print(
        "\nLoading evaluation datasets..."
    )

    forget_raw = read_file(
        cfg0.forget_path
    )

    if "type" in forget_raw.columns:

        forget_raw = (
            forget_raw.loc[
                forget_raw["type"]
                == "forget"
            ]
        )

    test_raw = read_file(
        cfg0.test_path
    )

    audit_raw = read_file(
        cfg0.audit_path
    )

    forget_data = (
        make_template_format(
            forget_raw,
            MODEL_NAME,
        )
    )

    # Keep same columns as your original
    # code when they exist.
    forget_columns = [
        "id",
        "question",
        "answer",
        "num_tokens",
        "type",
    ]

    existing_columns = [
        col
        for col in forget_columns
        if col in forget_data.columns
    ]

    forget_data = (
        forget_data[
            existing_columns
        ]
    )

    test_data = (
        make_template_format(
            test_raw,
            MODEL_NAME,
        )
    )

    audit_data = (
        make_template_format(
            audit_raw,
            MODEL_NAME,
        )
    )

    # -----------------------------------------------------
    # Embedding model
    # -----------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            cfg0.retriever_model,
            device=device,
        )
    )

    # -----------------------------------------------------
    # Base LLM
    # -----------------------------------------------------

    print(
        f"\nLoading base model: "
        f"{cfg0.model_id}"
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            cfg0.model_id,
            token=hf_token,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
            ),
        )
    )

    # -----------------------------------------------------
    # Bootstrap PEFT using first AVAILABLE adapter
    # -----------------------------------------------------

    first_cfg = runnable[0]

    first_adapter_name = (
        get_adapter_name(
            first_cfg
        )
    )

    print(
        "\nBootstrapping PEFT model with:"
    )

    print(
        first_cfg.adaptor_path
    )

    model = (
        PeftModel
        .from_pretrained(
            base_model,
            first_cfg.adaptor_path,
            adapter_name=(
                first_adapter_name
            ),
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
            ),
        )
    )

    model.eval()

    # -----------------------------------------------------
    # Evaluate everything available
    # -----------------------------------------------------

    for cfg in runnable:

        all_results = run_one(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            forget_data=forget_data,
            test_data=test_data,
            audit_data=audit_data,
            all_results=all_results,
            results_json_path=(
                results_json_path
            ),
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()