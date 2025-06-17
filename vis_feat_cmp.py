import os
from collections import defaultdict
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from lori_keystroke_features import all_ids, get_user_by_platform
from sklearn.metrics.pairwise import cosine_similarity
from rich.progress import track
import seaborn as sns


def pad_to_same_length(a, b, value=0.0):
    max_len = max(a.shape[1], b.shape[1])

    def pad(x):
        pad_width = max_len - x.shape[1]
        if pad_width > 0:
            return np.pad(
                x,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=value,
            )
        return x

    return pad(a), pad(b)


def unpack_list(lst):
    """
    Unpack a list of lists into a single list.
    """
    return list(itertools.chain(*lst))


def get_kht_data(i, enroll_platform_id, enroll_session_id=None):
    """
    Get the data for a specific user and platform/session.
    """
    df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)

    with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
        config = json.load(f)

    ignore_outliers = config.get("ignore_outliers", False)

    # Apply filters
    filtered_df = df[df["valid"]]
    if ignore_outliers and "outlier" in df.columns:
        filtered_df = filtered_df[filtered_df["outlier"] == False]
        # print(filtered_df)
        # Group by 'key1' and aggregate HL into lists
    kht_dict = filtered_df.groupby("key1")["HL"].apply(list).to_dict()
    # print(list(kht_dict.values()))
    return unpack_list(list(kht_dict.values()))


def pad_vector(vec, target_len):
    vec = np.array(vec, dtype=np.float32)
    if len(vec) < target_len:
        return np.pad(vec, (0, target_len - len(vec)), constant_values=0.0)
    return vec[:target_len]


def get_kit_data(i, enroll_platform_id, enroll_session_id=None):
    """
    Computes Key Interval Time (KIT) data from a given dataframe based on a specified feature type.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    - kit_feature_type (int): Specifies the type of KIT feature to compute. The valid values are:
      1: Time between release of the first key and press of the second key.
      2: Time between release of the first key and release of the second key.
      3: Time between press of the first key and press of the second key.
      4: Time between press of the first key and release of the second key.

    Returns:
    - dict: A dictionary where keys are pairs of consecutive key characters and values are lists containing
      computed KIT values based on the specified feature type for each instance of the key pair.

    Note:
    This function computes the KIT for each pair of consecutive keys in the dataframe and aggregates
    the results by key pair. The method for computing the KIT is determined by the `kit_feature_type` parameter.
    """
    kit_dict = defaultdict(list)
    with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
        config = json.load(f)
    ignore_outliers = config.get("ignore_outliers", False)
    df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)

    if df.empty:
        # print("dig deeper: dataframe is empty!")
        return kit_dict
    num_rows = len(df.index)
    for i in range(num_rows):
        if i < num_rows - 1:
            current_row = df.iloc[i]
            # print(type(current_row))
            if current_row.empty:
                print("dig deeper: row is empty!")
                return kit_dict
            if current_row["valid"] is False:
                continue
            if ignore_outliers and current_row["outlier"]:
                continue
            key = str(current_row["key1"]) + str(current_row["key2"])
            kit_dict[key].append(current_row["IL"])
    cleaned_dict = {}
    for feat, values in kit_dict.items():
        arr = np.array(values, dtype=np.float32)
        if np.all(np.isnan(arr)):
            continue  # Skip feature entirely if all are NaNs
        mean_val = np.nanmean(arr)
        arr[np.isnan(arr)] = mean_val
        cleaned_dict[feat] = arr.tolist()

    # Flatten all values from cleaned dict
    flattened = list(itertools.chain.from_iterable(cleaned_dict.values()))

    # Final safety: remove any leftover NaNs
    flattened = [x for x in flattened if not np.isnan(x)]

    return flattened


def plot_kht_data(d1, d2):
    """
    Plot the KHT data.
    """

    # Create x values for each series based on their length
    x1 = range(1, len(d1) + 1)  # x values for the first series
    x2 = range(1, len(d2) + 1)  # x values for the second series

    # Plotting the two series with different lengths
    plt.plot(x1, d1, label="Series 1", color="blue")
    plt.plot(x2, d2, label="Series 2", color="red")

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparing KHT Data from Two Platforms")

    # Adding a legend
    plt.legend()

    # Displaying the plot
    plt.show()


def plot_kht_cosine_similarity_heatmap():
    """
    Plot a heatmap of cosine similarity between KHT vectors of all users across platforms.
    """
    ids = all_ids()
    platform_ids = [1, 2, 3]
    vectors = []
    labels = []
    max_len = 0

    print("Loading all user-platform vectors...")
    for user_id in track(ids, description="Loading vectors"):
        for platform_id in platform_ids:
            kht = get_kht_data(user_id, platform_id)
            if not kht:
                continue
            labels.append(f"{user_id}_P{platform_id}")
            vectors.append(kht)
            max_len = max(max_len, len(kht))

    # === Pad all vectors to the same length ===
    print("Padding vectors...")
    padded_vectors = [pad_vector(v, max_len) for v in vectors]
    padded_vectors = np.stack(padded_vectors)

    # === Compute cosine similarity matrix ===
    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(padded_vectors)

    # === Plot ===
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sim_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", square=True
    )
    plt.title("Cross-User Cross-Platform Cosine Similarity (KHT)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("kht_cosine_similarity_heatmap.png")


def plot_kit_cosine_similarity_heatmap():
    """
    Plot a heatmap of cosine similarity between KHT vectors of all users across platforms.
    """
    ids = all_ids()
    platform_ids = [1, 2, 3]
    vectors = []
    labels = []
    max_len = 0

    print("Loading all user-platform vectors...")
    for user_id in track(ids, description="Loading vectors"):
        for platform_id in platform_ids:
            kht = get_kit_data(user_id, platform_id)
            if not kht:
                continue
            labels.append(f"{user_id}_P{platform_id}")
            vectors.append(kht)
            max_len = max(max_len, len(kht))

    # === Pad all vectors to the same length ===
    print("Padding vectors...")
    padded_vectors = [pad_vector(v, max_len) for v in vectors]
    padded_vectors = np.stack(padded_vectors)

    # === Compute cosine similarity matrix ===
    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(padded_vectors)

    # === Plot ===
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sim_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", square=True
    )
    plt.title("Cross-User Cross-Platform Cosine Similarity (KIT)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("kit_cosine_similarity_heatmap.png")


if __name__ == "__main__":
    plot_kht_cosine_similarity_heatmap()
    plot_kit_cosine_similarity_heatmap()
