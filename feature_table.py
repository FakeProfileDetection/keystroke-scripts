import os
import re
import json
import enum
import statistics
import random
import ast
from collections import defaultdict
import pandas as pd
import numpy as np
from alpha_bigrams_gen import alpha_word_bigrams
from lori_keystroke_features import (
    all_ids,
    create_kht_data_from_df,
    create_kit_data_from_df,
    get_user_by_platform,
    read_compact_format,
)
from rich.progress import track

with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)


class CKP_SOURCE(enum.Enum):
    FAKE_PROFILE_DATASET = 0
    ALPHA_WORDS = 1


def clean_string(s):
    # Remove extraneous single quotes and retain the actual content
    cleaned = re.sub(r"'\s*|\s*'", "", s)

    # Remove any extra spaces
    return cleaned.strip()


def columns_to_remove():
    return list(
        set(
            [
                "Unnamed: 0",
                "Unnamed: 126",
                "against",
                "methods",
                "could",
                "did",
                "remain",
                "realized",
                "teacher",
                "one-off",
                "realizes",
                "with ",  # NOTE: the space is intentional
                "can",
                "one-line",
                "slap",
                "sport",
                "and",
                "Rock",
                "that",
                "saying",
                "insensitive",
                "storyline",
                "behavior",
                "was",
                "does",
                "assault",
                ". He",  # NOTE: the space is intentional
                'wouldve"',
                'its"',
                ". It",
            ]
        )
    )


def map_platform_id_to_initial(platform_id: int):
    platform_mapping = {1: "f", 2: "i", 3: "t"}

    if platform_id not in platform_mapping:
        raise ValueError(f"Bad platform_id: {platform_id}")

    return platform_mapping[platform_id]


def flatten_list(nested_list):
    """
    Flattens a nested list into a single-level list.

    Args:
    nested_list (list): A list that may contain nested lists.

    Returns:
    list: A single-level list with all the elements from the nested lists.
    """

    return [
        item
        for sublist in nested_list
        for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])
    ]


def is_empty_list(x):
    return isinstance(x, list) and len(x) == 0


def is_evaluable(val):
    try:
        ast.literal_eval(val)
        return True
    except Exception:
        return False


def get_ckps(ckp_source: CKP_SOURCE):
    if ckp_source == CKP_SOURCE.FAKE_PROFILE_DATASET:
        return most_common_kepairs()
    elif ckp_source == CKP_SOURCE.ALPHA_WORDS:
        return alpha_word_bigrams()


def most_common_kepairs(n=10):
    freq = {}
    df = read_compact_format()
    kit1 = create_kit_data_from_df(df, 1)
    k = list(kit1.keys())
    for key in k:
        if key not in freq:
            freq[key] = len(kit1[key])
    # Sort the dictionary items by frequency in descending order
    sorted_items = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    # Extract the top 'n' keys from the sorted list
    return [key for key, _ in sorted_items[:n]]


def compute_fixed_feature_values(lst):
    if isinstance(lst, float):
        return [
            lst,
            lst,
            lst,
            lst,
            lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
            # lst,
        ]

    if len(lst) < 2:
        # This is just a way to make all of the KIT feature columns have the same length at the end
        # we can revert this back to just return the single element if we want to
        # return [lst[0], lst[0], lst[0], lst[0], lst[0]]
        return [
            lst[0],
            lst[0],
            lst[0],
            lst[0],
            lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
            # lst[0],
        ]

    # Convert to numpy array for convenience
    arr = np.array(lst)
    # Return the statistics as a list
    return [
        # np.min(arr),
        # np.max(arr),
        np.median(arr),
        np.mean(arr),
        np.std(arr),
        np.quantile(arr, 0.25),  # 1st quartile
        np.quantile(arr, 0.75),  # 3rd quartile
        # np.quantile(arr, 0.75) - np.quantile(arr, 0.25),  # IQR
        # stats.skew(arr),  # Skew
        # stats.kurtosis(arr),  # Kurtosis
        # np.max(arr) - np.min(arr),  # Range
        # np.std(arr) / np.mean(arr),  # Coefficient of Variation
        # np.var(arr),  # Variance
        # stats.entropy(np.histogram(arr, bins=10)[0]),  # Entropy
        # np.sqrt(np.mean(arr**2)),  # Root Mean Square
        # np.sum(arr**2),  # Energy
        # np.mean(arr) / np.std(arr),  # Signal-to-Noise Ratio
    ]


def create_full_user_and_platform_table(source: CKP_SOURCE):
    rows = []
    for i in track(all_ids()):
        for j in range(1, 4):
            for k in (1, 2):
                df = get_user_by_platform(i, j, k)
                if df.empty:
                    print(
                        f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}"
                    )
                    continue
                print(
                    f"User: {i}, platform: {map_platform_id_to_initial(j)}, session: {k}"
                )
                table = KeystrokeFeatureTable()
                # print(df.head())
                table.find_kit_from_most_common_keypairs(df, source)
                # print(list(table.inner.keys()))
                print(len(list(table.inner.keys())))
                # input()
                if config["use_kht_in_table"]:
                    table.find_kht_for_df(df)
                # TODO: do we still need to do this?
                # table.find_deft_for_df(df=df)
                table.add_user_platform_session_identifiers(i, j, k)

                row = table.as_df()
                # print(row)
                # input("Row to append")
                rows.append(row)
    return rows


def drop_empty_list_columns(df):
    # Identify columns where all values are empty lists
    columns_to_drop = [
        col for col in df.columns if df[col].apply(lambda x: x == []).all()
    ]

    # Drop these columns
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped


def flatten_kit_feature_columns(df: pd.DataFrame, ckps):
    cols = df.columns
    for col in cols:
        if col in ckps:
            df[col] = df[col].apply(lambda x: compute_fixed_feature_values(x))
    return df


def fill_empty_row_values(df: pd.DataFrame, ckps):
    cols = df.columns
    diffs = []
    for col in cols:
        if col in ckps:
            flat_data = flatten_list(list(df[col]))
            # TODO: Before calculating the mean remove all na values
            # TODO: check if this averaging across all users and if so we should just average across a single user
            data = statistics.mean([x for x in flat_data if not pd.isna(x)])
            # print(data)
            for element in flat_data:
                diffs.append(element - data)
            replacement_value = random.uniform(min(diffs), max(diffs))
            df[col] = df[col].apply(
                lambda x: (
                    replacement_value
                    if isinstance(x, float) and pd.isna(x)
                    else (
                        [replacement_value if pd.isna(i) else i for i in x]
                        if isinstance(x, list)
                        else x
                    )
                )
            )
        else:
            if col == "user_id" or col == "platform_id" or col == "session_id":
                pass
            else:
                raise ValueError(f"col: {col} is not in ckps")
    if config["use_kht_in_table"]:
        for col in df.columns:
            flat_data = flatten_list(list(df[col]))
            print(flat_data)
            # input("Flat data is")

            # Skip this column if flat_data contains NaN
            # TODO: I would rather us count the non-nan values and if there is at least 1 (that is a heuristic we can change)
            #       we just drop the nan's and keep the rest intact
            # TODO: looks like we also sometimes get flattened lists with integer values, that's not right - should debug
            #       We should do some typechecking to see if all flat_data is a list
            if any(not pd.isna(val) for val in flat_data):
                cleaned_data = [x for x in flat_data if not np.isnan(x)]
                data = statistics.mean(cleaned_data)
                print(data)

                # Calculate differences
                for element in cleaned_data:
                    diffs.append(element - data)

                # Generate a random replacement value
                replacement_value = random.uniform(min(diffs), max(diffs))

                # Modify the DataFrame column by replacing it with the calculated value
                # TODO: Instead of replacing all of the values, maybe we keep any existing values
                #       because they are used in the calculation anyway and then only add replacements
                #       till the list length is 5
                df[col] = df[col].apply(
                    lambda x: (
                        [replacement_value] * 5
                        if isinstance(x, list) and not x or not is_evaluable(str(x))
                        else x
                    )
                )
            else:
                print(
                    f"Skipping column {col} because it contains NaN values in flat_data."
                )
                continue

    # Remove columns where flat_data contained NaN
    df = df.dropna(axis=1, how="any")
    return df


def table_to_cleaned_df(table, source: CKP_SOURCE):
    combined_df = pd.concat(table, axis=0)
    print(combined_df)
    empty_list_count = combined_df.stack().map(is_empty_list).sum()
    nan_count = combined_df.isna().sum().sum()
    print(f"Number of cells containing empty lists: {empty_list_count}")
    print(f"Number of cells containing nans: {nan_count}")
    print(len(combined_df.columns))
    print(combined_df.columns)
    combined_df.to_csv("before_cleaning.csv")
    full_df = fill_empty_row_values(combined_df, get_ckps(source))
    empty_list_count = full_df.stack().map(is_empty_list).sum()
    nan_count = full_df.isna().sum().sum()
    print(f"Number of cells containing empty lists (post fill): {empty_list_count}")
    print(f"Number of cells containing nans (post fill): {nan_count}")
    print(len(full_df.columns))
    print(full_df.columns)
    full_df.to_csv("post_fill.csv")
    input()
    fixed_df = flatten_kit_feature_columns(full_df, get_ckps(source))
    cleaned = drop_empty_list_columns(fixed_df)
    return cleaned


class KeystrokeFeatureTable:
    def __init__(self) -> None:
        self.inner = defaultdict(list)

    # NOTE: We should not use the most common keypairs for the deft features because they rely on the distances between keys rather
    # than timing differences so all users may show up the same but I have to check
    def find_kit_from_most_common_keypairs(self, df, ckp_source: CKP_SOURCE):
        """
        Populate self.inner with KIT values for the most common key-pairs.
        """
        common_keypairs = get_ckps(ckp_source)

        for ckp in common_keypairs:
            for i in range(1, 5):
                # build the full KIT dict once
                kit_dict = create_kit_data_from_df(df, i)
                key = clean_string(ckp)

                # Debug: show whether this keypair exists
                exists = key in kit_dict
                print(f"[Feature {i}] {ckp} in keys? {exists}")
                print("Keypair:", ckp)
                if not exists:
                    print(f"DBG: KIT keypair is empty for {ckp} (feature {i})")
                    continue

                # extract the list of intervals
                kit_data = kit_dict[key]

                # Debug: inspect with a separate DataFrame
                debug_df = pd.DataFrame(kit_data, columns=[f"interval_{i}"])
                print(debug_df.to_string())

                # populate inner
                if len(kit_data) >= 1:
                    for entry in kit_data:
                        self.inner[ckp].append(entry)
                else:
                    print("DBG: KIT keypair in unexpected state:", kit_data)
                    raise NotImplementedError("Weird KIT state")

    def find_kht_for_df(self, df):
        kht_data = create_kht_data_from_df(df)
        for key in list(kht_data.keys()):
            # NOTE: This is a bit of a hack so that we can guarantee that kht data is of a
            #       fixed size like KIT
            #       The reason I have to do it here and can't write a similar function to flatten_kit is
            #       because I will have no way to check for kht specifically since KIT is comparing ckp list
            self.inner[key] = compute_fixed_feature_values(list(kht_data[key]))

    def get_raw(self):
        return self.inner

    def add_user_platform_session_identifiers(self, user_id, platform, session_id):
        self.inner["user_id"] = user_id
        self.inner["platform_id"] = platform
        if session_id is not None:
            self.inner["session_id"] = session_id

    def as_df(self):
        data = {key: [] for key in self.inner}
        for key, values in self.inner.items():
            if isinstance(values, list):
                data[key].append(values)
            else:
                data[key].append([values])

        return pd.DataFrame(data)


if __name__ == "__main__":
    source = CKP_SOURCE.FAKE_PROFILE_DATASET
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)
    cleaned.to_csv("fp_features_data.csv", mode="w+")
