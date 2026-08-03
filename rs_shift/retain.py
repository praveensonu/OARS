import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from pathlib import Path
from tqdm import tqdm


def select_retain_by_clustered_relative_shift(
    df: pd.DataFrame,
    *,
    retain_size: int,
    n_clusters: int,
    pre_representation_column: str = "h_pre",
    relative_shift_column: str = "r_i",
    type_column: str = "type",
    excluded_types: tuple[str, ...] = ("seed", "forget"),
    cluster_column: str = "retain_cluster",
    random_state: int = 42,
    kmeans_n_init: int | str = 10,
) -> pd.DataFrame:
    """
    Select retain samples using clustered pre-unlearning representations.

    Steps:
    1. Remove seed and forget samples.
    2. L2-normalize h_pre.
    3. Cluster normalized h_pre into K clusters.
    4. Sort each cluster by descending r_i.
    5. Select exactly retain_size / n_clusters samples per cluster.
    """
    if retain_size <= 0:
        raise ValueError("retain_size must be greater than zero.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than zero.")

    if retain_size % n_clusters != 0:
        raise ValueError(
            "retain_size must be exactly divisible by n_clusters. "
            f"Received retain_size={retain_size}, "
            f"n_clusters={n_clusters}."
        )

    required_columns = {
        pre_representation_column,
        relative_shift_column,
        type_column,
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    working_df = df.copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 1. Remove seed and forget samples
    # ---------------------------------------------------------

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
            ~normalized_types.isin(excluded_types_normalized)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(retain_pool) == 0:
        raise ValueError(
            "No samples remain after excluding seed and forget samples."
        )

    if n_clusters > len(retain_pool):
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the retain pool size "
            f"of {len(retain_pool)}."
        )

    if retain_size > len(retain_pool):
        raise ValueError(
            f"retain_size={retain_size} exceeds the retain pool size "
            f"of {len(retain_pool)}."
        )

    # ---------------------------------------------------------
    # 2. Validate r_i
    # ---------------------------------------------------------

    retain_pool[relative_shift_column] = pd.to_numeric(
        retain_pool[relative_shift_column],
        errors="coerce",
    )

    if not np.isfinite(
        retain_pool[relative_shift_column].to_numpy(
            dtype=np.float32
        )
    ).all():
        raise ValueError(
            f"{relative_shift_column!r} contains NaN or infinite values."
        )

    # ---------------------------------------------------------
    # 3. Stack and L2-normalize h_pre
    # ---------------------------------------------------------

    try:
        h_pre = np.stack(
            retain_pool[
                pre_representation_column
            ].to_numpy()
        ).astype(np.float32)
    except ValueError as error:
        raise ValueError(
            f"Could not stack {pre_representation_column!r}. "
            "All vectors must have the same dimension."
        ) from error

    if h_pre.ndim != 2:
        raise ValueError(
            "Pre-unlearning representations must form a 2D matrix."
        )

    if not np.isfinite(h_pre).all():
        raise ValueError(
            f"{pre_representation_column!r} contains NaN or inf."
        )

    pre_norms = np.linalg.norm(
        h_pre,
        ord=2,
        axis=1,
    )

    if np.any(pre_norms == 0):
        raise ValueError(
            "At least one pre-unlearning representation has zero norm."
        )

    h_pre_normalized = normalize(
        h_pre,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # 4. K-means clustering
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=kmeans_n_init,
    )

    retain_pool[cluster_column] = kmeans.fit_predict(
        h_pre_normalized
    ).astype(np.int32)

    # ---------------------------------------------------------
    # 5. Select equal number from every cluster
    # ---------------------------------------------------------

    samples_per_cluster = retain_size // n_clusters

    selected_clusters = []

    for cluster_id in range(n_clusters):
        cluster_df = (
            retain_pool.loc[
                retain_pool[cluster_column] == cluster_id
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
                f"Cluster {cluster_id} contains only "
                f"{len(cluster_df)} samples, but "
                f"{samples_per_cluster} are required."
            )

        selected_cluster = (
            cluster_df
            .head(samples_per_cluster)
            .copy()
        )

        selected_cluster[
            "cluster_selection_rank"
        ] = np.arange(
            1,
            samples_per_cluster + 1,
            dtype=np.int64,
        )

        selected_clusters.append(selected_cluster)

    retain_df = pd.concat(
        selected_clusters,
        axis=0,
        ignore_index=True,
    )

    retain_df = (
        retain_df
        .sort_values(
            by=[cluster_column, relative_shift_column],
            ascending=[True, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(retain_df) != retain_size:
        raise RuntimeError(
            f"Selected {len(retain_df)} samples instead of "
            f"{retain_size}."
        )

    return retain_df




MODELS = ["olmo"] #"qwen",
LOSSES = ["ga", "npo", "snpo"]

DATASET = "bio"

INPUT_DIR = Path(
    "/home/praveen/eacl/rs_shift/last_token"
)

OUTPUT_DIR = Path(
    "/home/praveen/eacl/data/retain"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RETAIN_SIZE = 200
N_CLUSTERS = 20


total_runs = len(MODELS) * len(LOSSES)
run_idx = 0

for model_name in MODELS:
    for current_loss in tqdm(LOSSES, desc=f"{model_name}"):
        run_idx += 1
        input_path = INPUT_DIR / f"{model_name}_{current_loss}_{DATASET}_shift.parquet"
        output_path = OUTPUT_DIR / f"{model_name}_{current_loss}_{DATASET}_retain.parquet"

        print(f"\n[{run_idx}/{total_runs}] Processing: {input_path.name}", flush=True)

        if not input_path.exists():
            print(f"  File not found, skipping: {input_path}", flush=True)
            continue

        shift_df = pd.read_parquet(input_path)
        print(f"  Loaded {len(shift_df)} rows.", flush=True)
        shift_df = shift_df[shift_df["type"] != "forget"]
        shift_df = shift_df[shift_df["type"] != "seed"]
        print(f"  {len(shift_df)} rows after filtering forget/seed.", flush=True)

        retain_df = select_retain_by_clustered_relative_shift(
                    shift_df,
                    retain_size=RETAIN_SIZE,
                    n_clusters=N_CLUSTERS,
                    pre_representation_column="h_pre",
                    relative_shift_column="r_i",
                    type_column="type",
                    excluded_types=("seed", "forget"),
                    cluster_column="retain_cluster",
                    random_state=42,
                    kmeans_n_init=10,
                )

        retain_df.to_parquet(output_path, index=False)
        print(f"  [{run_idx}/{total_runs}] Saved {len(retain_df)} retain samples to: {output_path}", flush=True)