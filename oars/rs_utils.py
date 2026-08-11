from __future__ import annotations

import gc
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_model_with_adapter(
    base_model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    *,
    use_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
):
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            torch_dtype = torch.float32

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map if torch.cuda.is_available() else None,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if use_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit loading requires CUDA.")

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )

    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_name_or_path,
            is_trainable=False,
        )

    model.eval()
    model.config.use_cache = False

    return model


def unload_model(model) -> None:
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.inference_mode()
def extract_last_token_representations(
    df: pd.DataFrame,
    model,
    tokenizer,
    *,
    prompt_formatter: Callable[[str], str],
    question_column: str = "question",
    answer_column: str = "answer",
    representation_column: str = "h_post",
    batch_size: int = 4,
    max_length: int = 512,
    storage_dtype: str = "float32",
) -> pd.DataFrame:
    """
    Extract raw last-token representations from the penultimate layer.

    The sequence is:

        formatted_prompt + answer

    EOS is deliberately not appended.

    The selected vector is:

        outputs.hidden_states[-2][sample, last_valid_token, :]

    No pooling or normalization is applied.
    """
    required_columns = {question_column, answer_column}
    missing = required_columns.difference(df.columns)

    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(
            "storage_dtype must be 'float16' or 'float32'."
        )

    out = df.copy().reset_index(drop=True)

    representations: list[list[float]] = []
    sequence_lengths: list[int] = []
    truncation_flags: list[bool] = []

    device = _input_device(model)

    for start in tqdm(
        range(0, len(out), batch_size),
        desc=f"Extracting {representation_column}",
    ):
        batch_df = out.iloc[start:start + batch_size]

        texts = []

        for _, row in batch_df.iterrows():
            prompt = prompt_formatter(str(row[question_column]))
            answer = str(row[answer_column])

            # Teacher-forced prompt + answer.
            # No EOS is appended.
            texts.append(prompt + answer)

        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
            return_length=False,
        )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        # [-1] is the final layer.
        # [-2] is the penultimate layer.
        hidden = outputs.hidden_states[-2]

        # Because padding is on the right:
        # final valid token index = sequence length - 1
        last_token_indices = attention_mask.sum(dim=1) - 1

        if torch.any(last_token_indices < 0):
            raise RuntimeError(
                "At least one sample contains zero valid tokens."
            )

        batch_indices = torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )

        last_token_hidden = hidden[
            batch_indices,
            last_token_indices.to(hidden.device),
            :,
        ]

        # Float conversion is only for dataframe storage.
        # No vector normalization is applied.
        representations_np = (
            last_token_hidden
            .float()
            .cpu()
            .numpy()
        )

        if storage_dtype == "float16":
            representations_np = representations_np.astype(
                np.float16
            )
        else:
            representations_np = representations_np.astype(
                np.float32
            )

        representations.extend(
            vector.tolist()
            for vector in representations_np
        )

        lengths = attention_mask.sum(dim=1).cpu().numpy()

        sequence_lengths.extend(
            int(length)
            for length in lengths
        )

        # Conservative flag: a sequence exactly equal to max_length
        # may or may not have originally been longer.
        truncation_flags.extend(
            bool(length >= max_length)
            for length in lengths
        )

        del (
            outputs,
            hidden,
            last_token_hidden,
            input_ids,
            attention_mask,
        )

    out[representation_column] = representations
    out[f"{representation_column}_sequence_tokens"] = sequence_lengths
    out[f"{representation_column}_was_truncated"] = truncation_flags

    return out


def compute_relative_shift_scores(
    df: pd.DataFrame,
    *,
    pre_representation_column: str = "h_pre",
    post_representation_column: str = "h_post",
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute the relative representation-shift score for every sample:

        v_i = (h_post_i - h_pre_i)
              / (||h_pre_i||_2 + epsilon)

        r_i = ||v_i||_2

    Equivalently:

        r_i = ||h_post_i - h_pre_i||_2
              / (||h_pre_i||_2 + epsilon)

    Returns a copy of the input DataFrame with the additional column
    ``r_i``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    required_columns = {
        pre_representation_column,
        post_representation_column,
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)

    h_pre = np.stack(
        out[pre_representation_column].to_numpy()
    ).astype(np.float32)

    h_post = np.stack(
        out[post_representation_column].to_numpy()
    ).astype(np.float32)

    if h_pre.ndim != 2 or h_post.ndim != 2:
        raise ValueError(
            "Representations must form two-dimensional arrays."
        )

    if h_pre.shape != h_post.shape:
        raise ValueError(
            "Pre/post representation shapes differ: "
            f"pre={h_pre.shape}, post={h_post.shape}"
        )

    if not np.all(np.isfinite(h_pre)):
        raise ValueError(
            f"{pre_representation_column!r} contains NaN or inf."
        )

    if not np.all(np.isfinite(h_post)):
        raise ValueError(
            f"{post_representation_column!r} contains NaN or inf."
        )

    eps = np.float32(epsilon)

    delta = h_post - h_pre

    pre_norm = np.linalg.norm(
        h_pre,
        ord=2,
        axis=1,
    ).astype(np.float32)

    v = delta / (pre_norm[:, None] + eps)

    r = np.linalg.norm(
        v,
        ord=2,
        axis=1,
    ).astype(np.float32)

    out["r_i"] = r

    return out




def extract_post_and_compute_scores(
    pre_df: pd.DataFrame,
    *,
    base_model_name_or_path: str,
    post_adapter_name_or_path: Optional[str],
    output_parquet_path: str,

    prompt_formatter: Callable[[str], str],
    pre_representation_column: str = "h_pre",
    post_representation_column: str = "h_post",
    question_column: str = "question",
    answer_column: str = "answer",
    batch_size: int = 4,
    max_length: int = 512,
    epsilon: float = 1e-12,
    use_4bit: bool = False,
    representation_storage_dtype: str = "float32",

) -> pd.DataFrame:
    """
    Extract post-unlearning representations and calculate directional scores.

    The input dataframe must already contain pre-unlearning representations.
    """
    required_columns = {
        question_column,
        answer_column,
        pre_representation_column,
    }

    missing = required_columns.difference(pre_df.columns)

    if missing:
        raise KeyError(
            f"Input dataframe is missing columns: {sorted(missing)}"
        )

    tokenizer = load_tokenizer(
        base_model_name_or_path
    )

    model = load_model_with_adapter(
        base_model_name_or_path,
        post_adapter_name_or_path,
        use_4bit=use_4bit,
    )

    try:
        post_df = extract_last_token_representations(
            pre_df,
            model,
            tokenizer,
            prompt_formatter=prompt_formatter,
            question_column=question_column,
            answer_column=answer_column,
            representation_column=post_representation_column,
            batch_size=batch_size,
            max_length=max_length,
            storage_dtype=representation_storage_dtype,
        )
    finally:
        unload_model(model)

    scored_df = compute_relative_shift_scores(
        post_df,
        pre_representation_column=pre_representation_column,
        post_representation_column=post_representation_column,
        epsilon=epsilon,
    )

    scored_df.to_parquet(
        output_parquet_path,
        index=False,
    )


    print(
        f"Saved {len(scored_df):,} scored samples to:\n"
        f"{output_parquet_path}"
    )

    return scored_df

def extract_and_save_pre_representations(
    df: pd.DataFrame,
    *,
    base_model_name_or_path: str,
    pre_adapter_name_or_path: Optional[str],
    output_parquet_path: str,
    prompt_formatter: Callable[[str], str],
    question_column: str = "question",
    answer_column: str = "answer",
    pre_representation_column: str = "h_pre",
    batch_size: int = 4,
    max_length: int = 512,
    use_4bit: bool = False,
    storage_dtype: str = "float32",
) -> pd.DataFrame:
    """
    Extract pre-unlearning raw last-token penultimate-layer representations.

    This should normally be run only once for each model/dataset pair.

    The extracted representation is:
        - teacher-forced prompt + answer
        - no EOS appended
        - last valid answer token
        - penultimate transformer layer
        - no pooling
        - no normalization
    """
    tokenizer = load_tokenizer(
        base_model_name_or_path
    )

    model = load_model_with_adapter(
        base_model_name_or_path,
        pre_adapter_name_or_path,
        use_4bit=use_4bit,
    )

    try:
        pre_df = extract_last_token_representations(
            df,
            model,
            tokenizer,
            prompt_formatter=prompt_formatter,
            question_column=question_column,
            answer_column=answer_column,
            representation_column=pre_representation_column,
            batch_size=batch_size,
            max_length=max_length,
            storage_dtype=storage_dtype,
        )
    finally:
        unload_model(model)

    pre_df.to_parquet(
        output_parquet_path,
        index=False,
    )

    print(
        f"Saved {len(pre_df):,} pre-unlearning representations to:\n"
        f"{output_parquet_path}"
    )

    return pre_df


def select_retain_by_clustered_relative_shift(
    df,
    *,
    retain_size,
    n_clusters,
    pre_representation_column="h_pre",
    relative_shift_column="r_i",
    type_column="type",
    excluded_types=("seed", "forget"),
    cluster_column="retain_cluster",
    random_state=42,
    kmeans_n_init=10,
):

    import numpy as np

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if retain_size <= 0:
        raise ValueError(
            "retain_size must be greater than zero."
        )

    if n_clusters <= 0:
        raise ValueError(
            "n_clusters must be greater than zero."
        )

    if retain_size % n_clusters != 0:

        raise ValueError(
            "retain_size must be exactly divisible "
            "by n_clusters. "
            f"retain_size={retain_size}, "
            f"n_clusters={n_clusters}"
        )

    required_columns = {
        pre_representation_column,
        relative_shift_column,
        type_column,
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:

        raise KeyError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    working_df = (
        df.copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Remove seed + forget
    # -----------------------------------------------------

    excluded_types_normalized = {
        str(value).strip().lower()
        for value in excluded_types
    }

    normalized_types = (
        working_df[type_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    retain_pool = (
        working_df.loc[
            ~normalized_types.isin(
                excluded_types_normalized
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(retain_pool) == 0:

        raise ValueError(
            "No samples remain after excluding "
            "seed and forget samples."
        )

    if n_clusters > len(retain_pool):

        raise ValueError(
            f"n_clusters={n_clusters} exceeds "
            f"retain pool size={len(retain_pool)}."
        )

    if retain_size > len(retain_pool):

        raise ValueError(
            f"retain_size={retain_size} exceeds "
            f"retain pool size={len(retain_pool)}."
        )

    # -----------------------------------------------------
    # Validate r_i
    # -----------------------------------------------------

    retain_pool[
        relative_shift_column
    ] = pd.to_numeric(
        retain_pool[relative_shift_column],
        errors="coerce",
    )

    if not np.isfinite(
        retain_pool[
            relative_shift_column
        ].to_numpy(dtype=np.float32)
    ).all():

        raise ValueError(
            f"{relative_shift_column} contains "
            "NaN or infinite values."
        )

    # -----------------------------------------------------
    # Stack h_pre
    # -----------------------------------------------------

    try:

        h_pre = np.stack(
            retain_pool[
                pre_representation_column
            ].to_numpy()
        ).astype(
            np.float32
        )

    except ValueError as error:

        raise ValueError(
            f"Could not stack "
            f"{pre_representation_column}. "
            "All vectors must have equal dimension."
        ) from error

    if h_pre.ndim != 2:

        raise ValueError(
            "Pre-unlearning representations "
            "must form a 2D matrix."
        )

    if not np.isfinite(h_pre).all():

        raise ValueError(
            f"{pre_representation_column} "
            "contains NaN or inf."
        )

    pre_norms = np.linalg.norm(
        h_pre,
        ord=2,
        axis=1,
    )

    if np.any(pre_norms == 0):

        raise ValueError(
            "At least one pre representation "
            "has zero norm."
        )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    h_pre_normalized = normalize(
        h_pre,
        norm="l2",
        axis=1,
    ).astype(
        np.float32
    )

    # -----------------------------------------------------
    # K-means
    # -----------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=kmeans_n_init,
    )

    retain_pool[
        cluster_column
    ] = kmeans.fit_predict(
        h_pre_normalized
    ).astype(
        np.int32
    )

    # -----------------------------------------------------
    # Equal selection per cluster
    # -----------------------------------------------------

    samples_per_cluster = (
        retain_size // n_clusters
    )

    selected_clusters = []

    for cluster_id in range(
        n_clusters
    ):

        cluster_df = (
            retain_pool.loc[
                retain_pool[
                    cluster_column
                ] == cluster_id
            ]
            .sort_values(
                relative_shift_column,
                ascending=False,
                kind="mergesort",
            )
            .copy()
        )

        if len(cluster_df) < samples_per_cluster:

            raise ValueError(
                f"Cluster {cluster_id} contains "
                f"{len(cluster_df)} samples, "
                f"but {samples_per_cluster} "
                "are required."
            )

        selected_cluster = (
            cluster_df
            .head(
                samples_per_cluster
            )
            .copy()
        )

        selected_cluster[
            "cluster_selection_rank"
        ] = np.arange(
            1,
            samples_per_cluster + 1,
            dtype=np.int64,
        )

        selected_clusters.append(
            selected_cluster
        )

    retain_df = pd.concat(
        selected_clusters,
        axis=0,
        ignore_index=True,
    )

    retain_df = (
        retain_df
        .sort_values(
            by=[
                cluster_column,
                relative_shift_column,
            ],
            ascending=[
                True,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(retain_df) != retain_size:

        raise RuntimeError(
            f"Selected {len(retain_df)} "
            f"samples instead of "
            f"{retain_size}."
        )

    return retain_df