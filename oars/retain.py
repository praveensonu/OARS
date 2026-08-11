import argparse
import os
from functools import partial
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from tabulate import tabulate

from dotenv import load_dotenv

from config import Config_oars

from rs_utils import *


load_dotenv()
hf_key = os.getenv("HF_TOKEN")


# =========================================================
# Chat templates
# =========================================================

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


olmo_chat_template = (
    """<|endoftext|><|user|>{question}<|assistant|>"""
)


# =========================================================
# Prompt formatting
# =========================================================

def format_prompt_prefix(
    question,
    model_name,
):

    if model_name == "llama":

        return LLAMA3_CHAT_TEMPLATE.format(
            question=question
        )

    elif model_name == "qwen":

        return qwen_chat_template.format(
            question=question
        )

    elif model_name == "olmo":

        return olmo_chat_template.format(
            question=question
        )

    return question


def get_prompt_formatter(model_name):

    return partial(
        format_prompt_prefix,
        model_name=model_name,
    )


# =========================================================
# Arguments
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Extract representation shifts and "
            "perform OARS retain selection"
        )
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
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "muse",
            "bio",
        ],
    )

    return parser.parse_args()



# =========================================================
# Check impairment experiments
# =========================================================

LOSS_TYPES = [
    "ga",
    "npo",
    "snpo",
]


def prepare_experiments(
    model_name,
    dataset,
):

    experiments = []

    for loss_type in LOSS_TYPES:

        cfg = Config_oars(
            model_name=model_name,
            dataset=dataset,
            loss_type=loss_type,
        )

        adapter_exists = (
            Path(
                cfg.post_adapter_path
            ).is_dir()
        )

        status = (
            "RUN"
            if adapter_exists
            else "SKIP"
        )

        experiments.append(
            {
                "cfg": cfg,
                "available": adapter_exists,
                "status": status,
            }
        )

    return experiments


# =========================================================
# Main
# =========================================================

def main():

    cli_args = parse_args()

    model_name = (
        cli_args.model_name
    )

    dataset = (
        cli_args.dataset
    )

    prompt_formatter = (
        get_prompt_formatter(
            model_name
        )
    )

    # -----------------------------------------------------
    # Base config
    # -----------------------------------------------------

    cfg0 = Config_oars(
        model_name=model_name,
        dataset=dataset,
    )
    print(
    tabulate(
        [
            ("Model", cfg0.model_name),
            ("Dataset", cfg0.dataset),
            ("Retain size", cfg0.retain_size),
            ("Number of clusters", cfg0.n_clusters),
            (
                "Samples per cluster",
                cfg0.retain_size // cfg0.n_clusters,
            ),
            ("Pre adapter", cfg0.pre_adapter_path),
            ("Input data", cfg0.input_path),
            (
                "Pre representation output",
                cfg0.pre_representation_path,
            ),
        ],
        headers=["Field", "Value"],
        tablefmt="github",
    )
)
    # -----------------------------------------------------
    # Make output directories
    # -----------------------------------------------------

    Path(
        cfg0.pre_representation_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "./data/retain_sets"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Check essential pre-unlearning files
    # -----------------------------------------------------

    if not Path(
        cfg0.input_path
    ).is_file():

        raise FileNotFoundError(
            f"Training data not found: "
            f"{cfg0.input_path}"
        )

    if not Path(
        cfg0.pre_adapter_path
    ).is_dir():

        raise FileNotFoundError(
            f"Pre-unlearning adapter "
            f"not found: "
            f"{cfg0.pre_adapter_path}"
        )

    # -----------------------------------------------------
    # Find available impairment experiments
    # -----------------------------------------------------

    experiments = (
        prepare_experiments(
            model_name=model_name,
            dataset=dataset,
        )
    )

    # -----------------------------------------------------
    # Show plan BEFORE loading models
    # -----------------------------------------------------

    plan = []

    for exp in experiments:

        cfg = exp["cfg"]

        plan.append(
            [
                cfg.model_name,
                cfg.loss_type,
                cfg.dataset,
                exp["status"],
                cfg.post_adapter_path,
                cfg.selection_output_path,
            ]
        )

    print(
        "\n\n"
        "============ OARS Selection Plan ============\n"
    )

    print(
        tabulate(
            plan,
            headers=[
                "Model",
                "Loss",
                "Dataset",
                "Status",
                "Impair Adapter",
                "OARS Output",
            ],
            tablefmt="github",
        )
    )

    runnable = [
        exp["cfg"]
        for exp in experiments
        if exp["available"]
    ]

    print(
        f"\nAvailable impairment models: "
        f"{len(runnable)} / "
        f"{len(experiments)}"
    )

    if not runnable:

        print(
            "\nNo impairment checkpoints "
            "are available."
        )

        print(
            "Nothing to select."
        )

        return

    print(
        "\nOARS will be created for:\n"
    )

    for cfg in runnable:

        print(
            f"  - {cfg.model_name} "
            f"+ {cfg.loss_type} "
            f"+ {cfg.dataset}"
        )

    # =====================================================
    # PRE-UNLEARNING REPRESENTATIONS
    # =====================================================

    print(
        "\n\n"
        "========================================"
    )

    print(
        "Extracting pre-unlearning representations"
    )

    print(
        "========================================\n"
    )

    print(
        f"Base model: "
        f"{cfg0.model_id}"
    )

    print(
        f"Pre adapter: "
        f"{cfg0.pre_adapter_path}"
    )

    print(
        f"Input data: "
        f"{cfg0.input_path}"
    )

    print(
        f"Output: "
        f"{cfg0.pre_representation_path}"
    )

    # -----------------------------------------------------
    # Reuse pre representations if already available
    # -----------------------------------------------------

    if Path(
        cfg0.pre_representation_path
    ).is_file():

        print(
            "\nPre-unlearning representations "
            "already exist."
        )

        print(
            "Reusing existing file."
        )

        pre_df = pd.read_parquet(
            cfg0.pre_representation_path
        )

    else:

        original_df = (
            pd.read_parquet(
                cfg0.input_path
            )
        )

        pre_df = (
            extract_and_save_pre_representations(
                original_df,

                base_model_name_or_path=(
                    cfg0.model_id
                ),

                pre_adapter_name_or_path=(
                    cfg0.pre_adapter_path
                ),

                output_parquet_path=(
                    cfg0.pre_representation_path
                ),

                prompt_formatter=(
                    prompt_formatter
                ),

                question_column="question",

                answer_column="answer",

                pre_representation_column=(
                    "h_pre"
                ),

                batch_size=(
                    cfg0.batch_size
                ),

                max_length=(
                    cfg0.max_length
                ),

                use_4bit=False,

                storage_dtype="float32",
            )
        )

    print(
        f"\nPre representations: "
        f"{len(pre_df)} samples"
    )

    # =====================================================
    # EACH IMPAIRMENT MODEL
    # =====================================================

    total = len(runnable)

    for idx, cfg in enumerate(
        runnable,
        start=1,
    ):

        print(
            "\n\n"
            "========================================"
        )

        print(
            f"[{idx}/{total}] "
            f"{cfg.model_name} + "
            f"{cfg.loss_type} + "
            f"{cfg.dataset}"
        )

        print(
            "========================================\n"
        )

        # -------------------------------------------------
        # Compute post-impair representation + r_i
        # -------------------------------------------------

        print(
            "Post-impair adapter:"
        )

        print(
            cfg.post_adapter_path
        )

        print(
            "\nShift output:"
        )

        print(
            cfg.shift_output_path
        )
        
        # Reuse shift file if it already exists
        if Path(
            cfg.shift_output_path
        ).is_file():

            print(
                "\nShift representations "
                "already exist."
            )

            print(
                "Reusing existing file."
            )

            scored_df = pd.read_parquet(
                cfg.shift_output_path
            )

        else:

            scored_df = (
                extract_post_and_compute_scores(
                    pre_df,

                    base_model_name_or_path=(
                        cfg.model_id
                    ),

                    post_adapter_name_or_path=(
                        cfg.post_adapter_path
                    ),

                    output_parquet_path=(
                        cfg.shift_output_path
                    ),

                    prompt_formatter=(
                        prompt_formatter
                    ),

                    pre_representation_column=(
                        "h_pre"
                    ),

                    post_representation_column=(
                        "h_post"
                    ),

                    question_column="question",

                    answer_column="answer",

                    batch_size=(
                        cfg.batch_size
                    ),

                    max_length=(
                        cfg.max_length
                    ),

                    epsilon=(
                        cfg.epsilon
                    ),

                    use_4bit=False,

                    representation_storage_dtype=(
                        "float32"
                    ),
                )
            )

        # =================================================
        # OARS selection
        # =================================================

        print(
            "\nSelecting OARS repair set..."
        )

        retain_df = (
            select_retain_by_clustered_relative_shift(
                scored_df,

                retain_size=(
                    cfg.retain_size
                ),

                n_clusters=(
                    cfg.n_clusters
                ),

                pre_representation_column=(
                    "h_pre"
                ),

                relative_shift_column=(
                    "r_i"
                ),

                type_column="type",

                excluded_types=(
                    "seed",
                    "forget",
                ),

                cluster_column=(
                    "retain_cluster"
                ),

                random_state=42,

                kmeans_n_init=10,
            )
        )

        # -------------------------------------------------
        # Save OARS selection
        # -------------------------------------------------

        retain_df.to_parquet(
            cfg.selection_output_path,
            index=False,
        )

        print(
            f"\nSelected "
            f"{len(retain_df)} samples."
        )

        print(
            "Saved OARS data to:"
        )

        print(
            cfg.selection_output_path
        )

    # =====================================================
    # Done
    # =====================================================

    print(
        "\n\n"
        "============ OARS COMPLETE ============\n"
    )

    summary = []

    for cfg in runnable:

        output_exists = Path(
            cfg.selection_output_path
        ).is_file()

        summary.append(
            [
                cfg.model_name,
                cfg.loss_type,
                cfg.dataset,
                (
                    "DONE"
                    if output_exists
                    else "FAILED"
                ),
                cfg.selection_output_path,
            ]
        )

    print(
        tabulate(
            summary,
            headers=[
                "Model",
                "Loss",
                "Dataset",
                "Status",
                "Output",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()