# select_bottom_gradient_similarity.py

from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


SUPPORTED_EXTENSIONS = {".pt", ".pth", ".bin"}


def natural_sort_key(path: Path):
    """
    Sort paths naturally:
        gradient_2.pt comes before gradient_10.pt
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def extract_tensor(obj: Any, tensor_key: Optional[str] = None) -> torch.Tensor:
    """
    Extract and flatten a gradient tensor from a loaded PyTorch object.

    Supported objects:
      1. A single tensor
      2. A dictionary containing one tensor
      3. A dictionary containing multiple parameter-gradient tensors
      4. A list/tuple of tensors

    If tensor_key is provided, obj[tensor_key] is used.
    """
    if tensor_key is not None:
        if not isinstance(obj, dict):
            raise TypeError(
                f"tensor_key='{tensor_key}' was provided, "
                f"but the loaded object is {type(obj)}."
            )

        if tensor_key not in obj:
            raise KeyError(
                f"Key '{tensor_key}' not found. "
                f"Available keys: {list(obj.keys())}"
            )

        obj = obj[tensor_key]

    if isinstance(obj, torch.Tensor):
        return obj.detach().reshape(-1)

    if isinstance(obj, dict):
        tensors = []

        # Sorting ensures that parameter gradients are concatenated
        # in the same order for every sample.
        for key in sorted(obj.keys()):
            value = obj[key]

            if isinstance(value, torch.Tensor):
                tensors.append(value.detach().reshape(-1))

        if not tensors:
            raise ValueError("Dictionary does not contain any tensors.")

        return torch.cat(tensors, dim=0)

    if isinstance(obj, (list, tuple)):
        tensors = [
            value.detach().reshape(-1)
            for value in obj
            if isinstance(value, torch.Tensor)
        ]

        if not tensors:
            raise ValueError("List/tuple does not contain any tensors.")

        return torch.cat(tensors, dim=0)

    raise TypeError(
        f"Unsupported saved gradient object type: {type(obj)}"
    )


def load_gradient(
    file_path: Path,
    tensor_key: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load one gradient file onto CPU and return a flat vector.
    """
    loaded = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )

    gradient = extract_tensor(loaded, tensor_key=tensor_key)

    if not gradient.is_floating_point():
        gradient = gradient.float()

    return gradient.to(dtype=dtype)


def get_gradient_files(folder: str | Path) -> list[Path]:
    """
    Recursively find supported PyTorch gradient files.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    files.sort(key=natural_sort_key)

    if not files:
        raise FileNotFoundError(
            f"No gradient files with extensions "
            f"{SUPPORTED_EXTENSIONS} found in {folder}"
        )

    return files


def get_id_from_filename(file_path: Path) -> str:
    """
    By default, use the complete filename stem as the sample ID.

    Examples:
        123.pt             -> "123"
        gradient_123.pt    -> "gradient_123"

    Modify this function if only part of your filename represents the ID.
    """
    return file_path.stem


@torch.inference_mode()
def build_mean_normalised_reference(
    reference_files: list[Path],
    device: torch.device,
    tensor_key: Optional[str] = None,
    gpu_dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, int]:
    """
    Construct:

        mean_reference = mean_j(reference_j / ||reference_j||_2)

    This produces exactly the same mean cosine score as calculating
    all pairwise cosine similarities and averaging over references.
    """
    mean_reference = None
    expected_dimension = None

    for file_path in tqdm(
        reference_files,
        desc="Building reference vector",
    ):
        gradient = load_gradient(
            file_path,
            tensor_key=tensor_key,
            dtype=torch.float32,
        )

        if expected_dimension is None:
            expected_dimension = gradient.numel()
        elif gradient.numel() != expected_dimension:
            raise ValueError(
                f"Gradient dimension mismatch in {file_path}.\n"
                f"Expected: {expected_dimension:,}\n"
                f"Found:    {gradient.numel():,}"
            )

        gradient = gradient.to(
            device=device,
            dtype=gpu_dtype,
            non_blocking=True,
        )

        norm = torch.linalg.vector_norm(gradient)

        if norm.item() <= eps:
            raise ValueError(
                f"Near-zero reference gradient found: {file_path}"
            )

        normalised_gradient = gradient / norm.clamp_min(eps)

        if mean_reference is None:
            mean_reference = torch.zeros_like(normalised_gradient)

        mean_reference.add_(normalised_gradient)

        del gradient, normalised_gradient

    mean_reference.div_(len(reference_files))

    return mean_reference, expected_dimension


@torch.inference_mode()
def find_bottom_k_candidates(
    candidate_files: list[Path],
    mean_reference: torch.Tensor,
    expected_dimension: int,
    bottom_k: int,
    device: torch.device,
    tensor_key: Optional[str] = None,
    gpu_dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> list[dict]:
    """
    Find candidates with the lowest mean cosine similarity.

    Uses a heap, so memory usage is O(k), rather than storing scores
    for every candidate.
    """
    if bottom_k <= 0:
        raise ValueError("bottom_k must be greater than zero.")

    bottom_k = min(bottom_k, len(candidate_files))

    # Python heapq is a min-heap.
    #
    # Store (-score, id, path), making the root correspond to the
    # largest actual score currently inside the bottom-k collection.
    heap: list[tuple[float, str, str]] = []

    skipped_zero_norm = 0

    for file_path in tqdm(
        candidate_files,
        desc="Scoring candidate gradients",
    ):
        gradient = load_gradient(
            file_path,
            tensor_key=tensor_key,
            dtype=torch.float32,
        )

        if gradient.numel() != expected_dimension:
            raise ValueError(
                f"Gradient dimension mismatch in {file_path}.\n"
                f"Expected: {expected_dimension:,}\n"
                f"Found:    {gradient.numel():,}"
            )

        gradient = gradient.to(
            device=device,
            dtype=gpu_dtype,
            non_blocking=True,
        )

        norm = torch.linalg.vector_norm(gradient)

        if norm.item() <= eps:
            skipped_zero_norm += 1
            continue

        # Exact mean cosine similarity:
        #
        # mean_j cosine(candidate, reference_j)
        # = normalised_candidate dot mean(normalised_references)
        score = torch.dot(
            gradient / norm.clamp_min(eps),
            mean_reference,
        ).float().item()

        sample_id = get_id_from_filename(file_path)
        item = (-score, sample_id, str(file_path))

        if len(heap) < bottom_k:
            heapq.heappush(heap, item)
        else:
            current_largest_bottom_score = -heap[0][0]

            if score < current_largest_bottom_score:
                heapq.heapreplace(heap, item)

        del gradient

    if skipped_zero_norm:
        print(
            f"Warning: skipped {skipped_zero_norm} candidate gradients "
            "because their norm was zero or nearly zero."
        )

    results = [
        {
            "id": sample_id,
            "mean_cosine_similarity": -negative_score,
            "gradient_path": gradient_path,
        }
        for negative_score, sample_id, gradient_path in heap
    ]

    results.sort(key=lambda row: row["mean_cosine_similarity"])

    return results


def align_id_type(
    selected_ids: pd.Series,
    dataframe_ids: pd.Series,
) -> pd.Series:
    """
    Convert selected filename IDs to the dataframe ID dtype where possible.
    """
    if pd.api.types.is_integer_dtype(dataframe_ids.dtype):
        return pd.to_numeric(selected_ids, errors="raise").astype(
            dataframe_ids.dtype
        )

    if pd.api.types.is_float_dtype(dataframe_ids.dtype):
        return pd.to_numeric(selected_ids, errors="raise").astype(
            dataframe_ids.dtype
        )

    return selected_ids.astype(str)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".jsonl", ".json"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(
        f"Unsupported dataframe format: {suffix}. "
        "Use parquet, CSV, JSON/JSONL, or pickle."
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return

    if suffix == ".csv":
        df.to_csv(path, index=False)
        return

    if suffix == ".jsonl":
        df.to_json(
            path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        return

    if suffix == ".json":
        df.to_json(
            path,
            orient="records",
            force_ascii=False,
            indent=2,
        )
        return

    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return

    raise ValueError(
        f"Unsupported output format: {suffix}. "
        "Use parquet, CSV, JSON/JSONL, or pickle."
    )


def select_dataframe_rows(
    dataframe_path: str | Path,
    selection_results: list[dict],
    dataframe_id_column: str,
) -> pd.DataFrame:
    df = load_dataframe(dataframe_path)

    if dataframe_id_column not in df.columns:
        raise KeyError(
            f"Column '{dataframe_id_column}' is not in the dataframe.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep IDs such as m1f1 as strings.
    df[dataframe_id_column] = (
        df[dataframe_id_column]
        .astype(str)
        .str.strip()
    )

    score_df = pd.DataFrame(selection_results)

    score_df["id"] = (
        score_df["id"]
        .astype(str)
        .str.strip()
    )

    if df[dataframe_id_column].duplicated().any():
        duplicated = (
            df.loc[
                df[dataframe_id_column].duplicated(keep=False),
                dataframe_id_column,
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Dataframe ID column '{dataframe_id_column}' is not unique. "
            f"Example duplicate IDs: {duplicated}"
        )

    selected_df = score_df.merge(
        df,
        left_on="id",
        right_on=dataframe_id_column,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = selected_df.loc[
        selected_df["_merge"] == "left_only",
        "id",
    ].tolist()

    if missing:
        raise KeyError(
            f"{len(missing)} selected gradient IDs were not found in "
            f"dataframe column '{dataframe_id_column}'.\n"
            f"First missing IDs: {missing[:20]}"
        )

    selected_df = selected_df.drop(columns="_merge")

    if dataframe_id_column != "id":
        selected_df = selected_df.drop(columns="id")

    return selected_df.sort_values(
        "mean_cosine_similarity",
        ascending=True,
    ).reset_index(drop=True)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Choose from {list(mapping)}."
        )

    return mapping[dtype_name]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select dataframe samples whose gradients have the lowest "
            "mean cosine similarity to a reference gradient set."
        )
    )

    parser.add_argument(
        "--reference-gradient-dir",
        required=True,
        help="Folder containing the approximately 100 reference gradients.",
    )
    parser.add_argument(
        "--candidate-gradient-dir",
        required=True,
        help="Folder containing the larger candidate gradient collection.",
    )
    parser.add_argument(
        "--dataframe-path",
        required=True,
        help="Input dataframe: parquet, CSV, JSON/JSONL, or pickle.",
    )
    parser.add_argument(
        "--dataframe-id-column",
        default="id",
        help="Dataframe column corresponding to gradient filename IDs.",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        required=True,
        help="Number of lowest-scoring candidate samples to select.",
    )
    parser.add_argument(
        "--output-dataframe",
        required=True,
        help="Path for the selected dataframe.",
    )
    parser.add_argument(
        "--output-scores",
        default=None,
        help="Optional path for saving selected IDs and scores as CSV.",
    )
    parser.add_argument(
        "--tensor-key",
        default=None,
        help=(
            "Optional dictionary key containing the gradient tensor, "
            "for example 'gradient' or 'grad'."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device, for example cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--gpu-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help=(
            "GPU calculation dtype. float32 is safest; float16 or "
            "bfloat16 uses less GPU memory."
        ),
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)
    gpu_dtype = parse_dtype(args.gpu_dtype)

    reference_files = get_gradient_files(
        args.reference_gradient_dir
    )
    candidate_files = get_gradient_files(
        args.candidate_gradient_dir
    )

    print(f"Reference gradients: {len(reference_files):,}")
    print(f"Candidate gradients: {len(candidate_files):,}")
    print(f"Device: {device}")
    print(f"GPU calculation dtype: {gpu_dtype}")

    mean_reference, gradient_dimension = (
        build_mean_normalised_reference(
            reference_files=reference_files,
            device=device,
            tensor_key=args.tensor_key,
            gpu_dtype=gpu_dtype,
        )
    )

    print(f"Gradient dimension: {gradient_dimension:,}")

    selection_results = find_bottom_k_candidates(
        candidate_files=candidate_files,
        mean_reference=mean_reference,
        expected_dimension=gradient_dimension,
        bottom_k=args.bottom_k,
        device=device,
        tensor_key=args.tensor_key,
        gpu_dtype=gpu_dtype,
    )

    selected_df = select_dataframe_rows(
        dataframe_path=args.dataframe_path,
        selection_results=selection_results,
        dataframe_id_column=args.dataframe_id_column,
    )

    save_dataframe(
        selected_df,
        args.output_dataframe,
    )

    score_output_path = (
        Path(args.output_scores)
        if args.output_scores
        else Path(args.output_dataframe).with_name(
            Path(args.output_dataframe).stem + "_scores.csv"
        )
    )

    score_output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(selection_results).to_csv(
        score_output_path,
        index=False,
    )

    metadata_path = Path(args.output_dataframe).with_name(
        Path(args.output_dataframe).stem + "_metadata.json"
    )

    metadata = {
        "reference_gradient_directory": str(
            Path(args.reference_gradient_dir).resolve()
        ),
        "candidate_gradient_directory": str(
            Path(args.candidate_gradient_dir).resolve()
        ),
        "number_of_reference_gradients": len(reference_files),
        "number_of_candidate_gradients": len(candidate_files),
        "gradient_dimension": gradient_dimension,
        "requested_bottom_k": args.bottom_k,
        "selected_samples": len(selected_df),
        "device": str(device),
        "gpu_dtype": str(gpu_dtype),
        "dataframe_id_column": args.dataframe_id_column,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSelection completed.")
    print(f"Selected dataframe: {args.output_dataframe}")
    print(f"Selected scores:    {score_output_path}")
    print(f"Metadata:           {metadata_path}")
    print("\nLowest scores:")
    print(
        pd.DataFrame(selection_results)
        .head(10)[["id", "mean_cosine_similarity"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()