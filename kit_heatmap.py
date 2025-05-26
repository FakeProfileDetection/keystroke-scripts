import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from feature_table import most_common_kepairs
from lori_keystroke_features import (
    read_compact_format,
    all_ids,
    create_kit_data_from_df,
)
import pandas as pd

def missing_common_keypairs():
    most_common_keys = most_common_kepairs()
    df = read_compact_format()
    results = []

    for i in track(all_ids()):
        for j in (1, 2, 3):
            for k in (1, 2):
                df_user = df[(df['user_id'] == i) & (df['platform_id'] == j) & (df['session_id'] == k)]
                if df_user.empty:
                    bads = np.nan  # Use NaN for missing data
                else:
                    l = list(create_kit_data_from_df(df_user, 1).keys())
                    bads = len([item for item in most_common_keys if item not in l])
                results.append({
                    "user_id": i,
                    "platform_id": j,
                    "session_id": k,
                    "bads": bads
                })

    # Create DataFrame
    df_bads = pd.DataFrame(results)
    # Pivot to wide format: users x (platform_session)
    mapping = {1: "F", 2: "I", 3: "T"}
    df_bads["col"] = df_bads.apply(lambda row: f"{mapping[row['platform_id']]}_{row['session_id']}", axis=1)
    pivot = df_bads.pivot(index="user_id", columns="col", values="bads")

    # Ensure platform_session ordering F_1, F_2, I_1, I_2, T_1, T_2
    col_order = [f"{m}_{s}" for m in ["F", "I", "T"] for s in [1, 2]]
    pivot = pivot.reindex(columns=col_order)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, pivot.shape[0] * 0.2)))
    heatmap = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add axes labels
    ax.set_xlabel("platform_session")
    ax.set_ylabel("user_id")

    fig.colorbar(heatmap, label="Missing Most Common Keypairs (bads)")
    ax.set_title("Heatmap of Missing Most Common Keypairs by User, Platform, Session")

    plt.tight_layout()
    plt.savefig("missing_keys_heatmap.png")
    plt.show()

def nan_count_heatmap():
    df = read_compact_format()
    grouped = (
        df.groupby(["user_id", "platform_id", "session_id"])
        .apply(lambda g: g.isna().sum().sum())
        .reset_index(name="nan_count")
    )

    # Pivot to wide format: users x (platform_session)
    pivot = grouped.pivot(index="user_id", columns=["platform_id", "session_id"], values="nan_count")

    # Rename columns to "F_1", "F_2", "I_1", etc.
    mapping = {1: "F", 2: "I", 3: "T"}
    pivot.columns = [f"{mapping[p]}_{s}" for p, s in pivot.columns]

    # Ensure platform_session ordering F_1, F_2, I_1, I_2, T_1, T_2
    col_order = [f"{m}_{s}" for m in ["F", "I", "T"] for s in [1, 2]]
    pivot = pivot.reindex(columns=col_order)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, pivot.shape[0] * 0.2)))
    heatmap = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")

    # Configure axes
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Colorbar and title
    fig.colorbar(heatmap, label="NaN Count")
    ax.set_title("Heatmap of NaN Counts by User, Platform, Session")

    plt.tight_layout()
    plt.savefig("nan_counts_heatmap.png")
if __name__ == "__main__":
    missing_common_keypairs()
    # nan_count_heatmap()