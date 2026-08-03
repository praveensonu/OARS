from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



def select_forget_and_retain_sets(
    scored_df: pd.DataFrame,
    *,
    forget_size: int,
    retain_size: int,
    n_clusters: int,
    pre_representation_column: str = "h_pre",
    b_column: str = "b_i",
    q_column: str = "q_i",
    type_column: str = "type",
    seed_label: str = "seed",
    random_state: int = 42,
    kmeans_n_init: int | str = 10,
    cluster_column: str = "retain_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select forget and retain sets from a scored training dataframe.

    Forget-set selection
    --------------------
    1. Remove seed samples from the candidate pool.
    2. Sort all remaining samples by descending b_i.
    3. Select the top `forget_size` samples.

    Retain-set selection
    --------------------
    1. Remove both seed samples and selected forget samples.
    2. L2-normalize their pre-unlearning representations.
    3. Apply K-means with `n_clusters`.
    4. Within each cluster, sort by descending q_i.
    5. Select approximately equal numbers from all clusters.
    6. If `retain_size` is not divisible by `n_clusters`, distribute
       the remaining slots across clusters.
    7. If a cluster does not contain enough samples, redistribute its
       unused quota to other clusters.

    Returns
    -------
    forget_df:
        Selected forget samples.

    retain_df:
        Selected retain samples.

    candidate_df:
        All non-seed samples with selection diagnostics, including
        forget/retain flags and retain-cluster assignments.
    """
    if forget_size <= 0:
        raise ValueError("forget_size must be greater than zero.")

    if retain_size <= 0:
        raise ValueError("retain_size must be greater than zero.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than zero.")

    required_columns = {
        pre_representation_column,
        b_column,
        q_column,
        type_column,
    }

    missing = required_columns.difference(scored_df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    out = scored_df.copy().reset_index(drop=True)

    # Internal stable row identifier.
    # This avoids depending on whether the original dataframe index is unique.
    out["_selection_row_id"] = np.arange(
        len(out),
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # 1. Keep seed samples completely outside both pools
    # ---------------------------------------------------------

    seed_mask = (
        out[type_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(seed_label.strip().lower())
    )

    candidate_df = (
        out.loc[~seed_mask]
        .copy()
        .reset_index(drop=True)
    )

    if len(candidate_df) == 0:
        raise ValueError(
            "No non-seed samples are available for selection."
        )

    if forget_size > len(candidate_df):
        raise ValueError(
            f"forget_size={forget_size} exceeds the number of "
            f"non-seed samples={len(candidate_df)}."
        )

    # Ensure scalar scores are valid.
    if candidate_df[b_column].isna().any():
        number_missing = int(
            candidate_df[b_column].isna().sum()
        )
        raise ValueError(
            f"{b_column!r} contains {number_missing} missing values "
            "among non-seed samples."
        )

    if not np.isfinite(
        candidate_df[b_column].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError(
            f"{b_column!r} contains NaN or infinite values."
        )

    # ---------------------------------------------------------
    # 2. Forget selection: top-n signed b_i scores
    # ---------------------------------------------------------

    forget_df = (
        candidate_df
        .sort_values(
            by=b_column,
            ascending=False,
            kind="mergesort",
        )
        .head(forget_size)
        .copy()
    )

    forget_df["selection_rank"] = np.arange(
        1,
        len(forget_df) + 1,
        dtype=np.int64,
    )

    forget_df["selected_as"] = "forget"

    forget_ids = set(
        forget_df["_selection_row_id"].tolist()
    )

    # ---------------------------------------------------------
    # 3. Retain candidate pool:
    #    non-seed minus selected forget samples
    # ---------------------------------------------------------

    retain_pool = (
        candidate_df.loc[
            ~candidate_df["_selection_row_id"].isin(forget_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if retain_size > len(retain_pool):
        raise ValueError(
            f"retain_size={retain_size} exceeds the remaining "
            f"candidate pool size={len(retain_pool)} after selecting "
            f"{forget_size} forget samples."
        )

    if n_clusters > len(retain_pool):
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the retain candidate "
            f"pool size={len(retain_pool)}."
        )

    if retain_pool[q_column].isna().any():
        number_missing = int(
            retain_pool[q_column].isna().sum()
        )
        raise ValueError(
            f"{q_column!r} contains {number_missing} missing values "
            "in the retain candidate pool."
        )

    if not np.isfinite(
        retain_pool[q_column].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError(
            f"{q_column!r} contains NaN or infinite values."
        )

    # ---------------------------------------------------------
    # 4. Stack and L2-normalize pre-unlearning representations
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
            "All representation vectors must have the same dimension."
        ) from error

    if h_pre.ndim != 2:
        raise ValueError(
            "Pre-unlearning representations must form a "
            "two-dimensional matrix."
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

    zero_norm_mask = pre_norms <= 0

    if zero_norm_mask.any():
        number_zero = int(zero_norm_mask.sum())
        raise ValueError(
            f"Found {number_zero} zero-norm pre-unlearning "
            "representations. They cannot be L2-normalized."
        )

    # Equivalent to h_pre / ||h_pre||_2 row-wise.
    h_pre_normalized = normalize(
        h_pre,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # 5. K-means clustering
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=kmeans_n_init,
    )

    cluster_labels = kmeans.fit_predict(
        h_pre_normalized
    )

    retain_pool[cluster_column] = cluster_labels.astype(
        np.int32
    )

    # Store representation norms as a diagnostic.
    retain_pool[
        f"{pre_representation_column}_norm"
    ] = pre_norms.astype(np.float32)

    # ---------------------------------------------------------
    # 6. Determine equal initial quotas
    # ---------------------------------------------------------

    base_quota = retain_size // n_clusters
    remainder = retain_size % n_clusters

    cluster_sizes = (
        retain_pool[cluster_column]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Start with an equal quota for every cluster.
    quotas = {
        cluster_id: base_quota
        for cluster_id in range(n_clusters)
    }

    # When m is not divisible by K, distribute one additional
    # position to some clusters.
    #
    # Clusters are prioritized by:
    #   1. having enough available samples
    #   2. highest maximum q_i
    cluster_priority = []

    for cluster_id in range(n_clusters):
        cluster_rows = retain_pool.loc[
            retain_pool[cluster_column] == cluster_id
        ]

        maximum_q = (
            float(cluster_rows[q_column].max())
            if len(cluster_rows) > 0
            else -np.inf
        )

        cluster_priority.append(
            (
                cluster_id,
                maximum_q,
                len(cluster_rows),
            )
        )

    cluster_priority.sort(
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    for cluster_id, _, _ in cluster_priority[:remainder]:
        quotas[cluster_id] += 1

    # A cluster cannot contribute more samples than it contains.
    initial_shortfall = 0

    for cluster_id in range(n_clusters):
        available = cluster_sizes.get(cluster_id, 0)

        if quotas[cluster_id] > available:
            initial_shortfall += (
                quotas[cluster_id] - available
            )
            quotas[cluster_id] = available

    # ---------------------------------------------------------
    # 7. Redistribute quotas from undersized clusters
    # ---------------------------------------------------------

    while initial_shortfall > 0:
        eligible_clusters = []

        for cluster_id in range(n_clusters):
            available = cluster_sizes.get(cluster_id, 0)
            unused_capacity = available - quotas[cluster_id]

            if unused_capacity <= 0:
                continue

            cluster_rows = retain_pool.loc[
                retain_pool[cluster_column] == cluster_id
            ].sort_values(
                q_column,
                ascending=False,
                kind="mergesort",
            )

            # Score for the next currently unselected sample.
            next_position = quotas[cluster_id]

            next_q = float(
                cluster_rows.iloc[next_position][q_column]
            )

            eligible_clusters.append(
                (
                    cluster_id,
                    next_q,
                    unused_capacity,
                )
            )

        if not eligible_clusters:
            raise RuntimeError(
                "Unable to redistribute retain quotas. "
                "The remaining clusters do not have sufficient capacity."
            )

        # Give the next slot to the cluster whose next unselected
        # sample has the largest q_i.
        eligible_clusters.sort(
            key=lambda item: (
                item[1],
                item[2],
            ),
            reverse=True,
        )

        selected_cluster_id = eligible_clusters[0][0]
        quotas[selected_cluster_id] += 1
        initial_shortfall -= 1

    if sum(quotas.values()) != retain_size:
        raise RuntimeError(
            "Internal quota-allocation error: allocated "
            f"{sum(quotas.values())} samples instead of "
            f"{retain_size}."
        )

    # ---------------------------------------------------------
    # 8. Select top q_i samples within each cluster
    # ---------------------------------------------------------

    selected_cluster_frames = []

    for cluster_id in range(n_clusters):
        cluster_df = (
            retain_pool.loc[
                retain_pool[cluster_column] == cluster_id
            ]
            .sort_values(
                by=q_column,
                ascending=False,
                kind="mergesort",
            )
            .copy()
        )

        cluster_quota = quotas[cluster_id]

        selected_from_cluster = (
            cluster_df
            .head(cluster_quota)
            .copy()
        )

        selected_from_cluster[
            "cluster_selection_rank"
        ] = np.arange(
            1,
            len(selected_from_cluster) + 1,
            dtype=np.int64,
        )

        selected_from_cluster[
            "cluster_selection_quota"
        ] = cluster_quota

        selected_cluster_frames.append(
            selected_from_cluster
        )

    retain_df = pd.concat(
        selected_cluster_frames,
        axis=0,
        ignore_index=True,
    )

    retain_df["selected_as"] = "retain"

    # Sort for readable output:
    # cluster first, then descending q_i within each cluster.
    retain_df = (
        retain_df
        .sort_values(
            by=[cluster_column, q_column],
            ascending=[True, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(retain_df) != retain_size:
        raise RuntimeError(
            f"Selected {len(retain_df)} retain samples, "
            f"but retain_size={retain_size}."
        )

    if retain_df["_selection_row_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate samples were selected into the retain set."
        )

    retain_ids = set(
        retain_df["_selection_row_id"].tolist()
    )

    overlap = forget_ids.intersection(retain_ids)

    if overlap:
        raise RuntimeError(
            f"Forget and retain selections overlap by "
            f"{len(overlap)} samples."
        )

    # ---------------------------------------------------------
    # 9. Add diagnostics to the candidate dataframe
    # ---------------------------------------------------------

    cluster_by_row_id = (
        retain_pool
        .set_index("_selection_row_id")[cluster_column]
        .to_dict()
    )

    candidate_df[cluster_column] = (
        candidate_df["_selection_row_id"]
        .map(cluster_by_row_id)
        .astype("Int64")
    )

    candidate_df["selected_for_forget"] = (
        candidate_df["_selection_row_id"].isin(forget_ids)
    )

    candidate_df["selected_for_retain"] = (
        candidate_df["_selection_row_id"].isin(retain_ids)
    )

    # Remove internal helper ID from exported selections.
    forget_df = forget_df.drop(
        columns=["_selection_row_id"]
    ).reset_index(drop=True)

    retain_df = retain_df.drop(
        columns=["_selection_row_id"]
    ).reset_index(drop=True)

    candidate_df = candidate_df.drop(
        columns=["_selection_row_id"]
    ).reset_index(drop=True)

    return forget_df, retain_df, candidate_df


def select_and_save_forget_retain_sets(
    scored_df: pd.DataFrame,
    *,
    model_name: str,
    current_loss: str,
    dataset: str,
    forget_size: int,
    retain_size: int,
    n_clusters: int,
    output_directory: str | Path = ".",
    pre_representation_column: str = "h_pre",
    b_column: str = "b_i",
    q_column: str = "q_i",
    type_column: str = "type",
    seed_label: str = "seed",
    random_state: int = 42,
    kmeans_n_init: int | str = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run forget/retain selection and save both selected dataframes.
    """
    forget_df, retain_df, candidate_df = (
        select_forget_and_retain_sets(
            scored_df,
            forget_size=forget_size,
            retain_size=retain_size,
            n_clusters=n_clusters,
            pre_representation_column=(
                pre_representation_column
            ),
            b_column=b_column,
            q_column=q_column,
            type_column=type_column,
            seed_label=seed_label,
            random_state=random_state,
            kmeans_n_init=kmeans_n_init,
        )
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    forget_output_path = output_directory / (
        f"{model_name}_{current_loss}_{dataset}_forget.parquet"
    )

    retain_output_path = output_directory / (
        f"{model_name}_{current_loss}_{dataset}_retain.parquet"
    )

    forget_df.to_parquet(
        forget_output_path,
        index=False,
    )

    retain_df.to_parquet(
        retain_output_path,
        index=False,
    )

    print(
        f"Saved {len(forget_df):,} forget samples to:\n"
        f"{forget_output_path}"
    )

    print(
        f"\nSaved {len(retain_df):,} retain samples to:\n"
        f"{retain_output_path}"
    )

    return forget_df, retain_df, candidate_df