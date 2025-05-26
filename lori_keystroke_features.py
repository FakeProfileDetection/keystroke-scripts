import os
import pandas as pd
import numpy as np
from rich.progress import track
from collections import defaultdict


def read_compact_format():
    df = pd.read_csv(
        # TODO: replaced with small.csv for testing, actual file is typenet_features.csv
        os.path.join(os.getcwd(), "dataset", "lori_typenet_features.csv"),
        dtype={
            "user_id": np.uint16,
            "platform_id": np.uint8,
            "video_id": np.uint8,
            "session_id": np.uint8,
            "sequence_id": np.uint8,
            "key1": str,
            "key2": str,
            "key1_press": np.float64,
            "key1_release": np.float64,
            "key2_press": np.float64,
            "key2_release": np.float64,
            "HL": np.float64,
            "IL": np.float64,
            "PL": np.float64,
            "RL": np.float64,
            "key1_timestamp": np.float64,
        },
    )
    return df


def all_ids():
    """
    Retrieve a list of IDs based on the gender type specified in the configuration file.

    This function reads a configuration file named 'classifier_config.json' located in the current
    working directory. It then extracts the gender type from the file and returns a corresponding list
    of IDs. The IDs are assigned based on predefined gender categories.

    Returns:
    - list[int]: A list of IDs corresponding to the specified gender type.

    Raises:
    - ValueError: If the gender type in the configuration file is unrecognized.

    Preconditions:
    - The 'classifier_config.json' file should be present in the current working directory.
    - The file should contain a valid JSON structure with a 'gender' field.
    """
    df = read_compact_format()
    # ids = list(set(df["user_id"].tolist()))
    ids = df["user_id"].unique().tolist()
    return ids


def create_kht_data_from_df(df):
    """
    Computes Key Hold Time (KHT) data from a given dataframe.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    Returns:
    - dict: A dictionary where keys are individual key characters and values are lists containing
      computed KHT values (difference between the release time and press time) for each instance of the key.

    Note:
    KHT is defined as the difference between the release time and the press time for a given key instance.
    This function computes the KHT for each key in the dataframe and aggregates the results by key.
    """
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key1"]].append(row["key1_release"] - row["key1_press"])
        kht_dict[row["key2"]].append(row["key2_release"] - row["key2_press"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type):
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
            key = str(current_row["key1"]) + str(current_row["key2"])
            initial_press = float(current_row["key1_press"])
            second_press = float(current_row["key2_press"])
            initial_release = float(current_row["key1_release"])
            second_release = float(current_row["key2_release"])
            if kit_feature_type == 1:
                kit_dict[key].append(second_press - initial_release)
            elif kit_feature_type == 2:
                kit_dict[key].append(second_release - initial_release)
            elif kit_feature_type == 3:
                kit_dict[key].append(second_press - initial_press)
            elif kit_feature_type == 4:
                kit_dict[key].append(second_release - initial_press)
    return kit_dict


def get_user_by_platform(user_id, platform_id, session_id=None):
    """
    Retrieve data for a given user and platform, with an optional session_id filter.

    Parameters:
    - user_id (int): Identifier for the user.
    - platform_id (int or list[int]): Identifier for the platform.
      If provided as a list, it should contain two integers specifying
      an inclusive range to search between.
    - session_id (int or list[int], optional): Identifier for the session.
      If provided as a list, it can either specify an inclusive range with
      two integers or provide multiple session IDs to filter by.

    Returns:
    - DataFrame: Filtered data matching the given criteria.

    Notes:
    - When providing a list for platform_id or session_id to specify a range,
      the order of the two integers does not matter.
    - When providing a list with more than two integers for session_id,
      it will filter by those exact session IDs.

    Raises:
    - AssertionError: If platform_id or session_id list does not follow the expected format.

    Examples:
    >>> df = get_user_by_platform(123, 1)
    >>> df = get_user_by_platform(123, [1, 5])
    >>> df = get_user_by_platform(123, 1, [2, 6])
    >>> df = get_user_by_platform(123, 1, [2, 3, 4])

    """
    # Get all of the data for a user amd platform with am optional session_id

    # print(f"user_id:{user_id}", end=" | ")
    df = read_compact_format()
    if session_id is None:
        if isinstance(platform_id, list):
            # Should only contain an inclusive range of the starting id and ending id
            assert len(platform_id) == 2
            if platform_id[0] < platform_id[1]:
                return df[
                    (df["user_id"] == user_id)
                    & (df["platform_id"].between(platform_id[0], platform_id[1]))
                ]
            else:
                return df[
                    (df["user_id"] == user_id)
                    & (df["platform_id"].between(platform_id[1], platform_id[0]))
                ]
        else:
            # print(df[df["user_id"] == user_id])
            # input()
            return df[(df["user_id"] == user_id) & (df["platform_id"] == platform_id)]
    if isinstance(session_id, list):
        # Should only contain an inclusive range of the starting id and ending id
        if len(session_id) == 2:
            return df[
                (df["user_id"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].between(session_id[0], session_id[1]))
            ]
        elif len(session_id) > 2:
            test = df[
                (df["user_id"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]
            # print(session_id)
            # print(test["session_id"].unique())
            # input()
            return df[
                (df["user_id"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]

    return df[
        (df["user_id"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


def map_platform_id_to_initial(platform_id: int):
    platform_mapping = {1: "f", 2: "i", 3: "t"}

    if platform_id not in platform_mapping:
        raise ValueError(f"Bad platform_id: {platform_id}")

    return platform_mapping[platform_id]


if __name__ == "__main__":
    for i in track(all_ids()):
        for j in range(1, 4):
            df = get_user_by_platform(i, j)
            if df.empty:
                print(f"Skipping User: {i}, platform: {map_platform_id_to_initial(j)}")
                continue
            print(f"User: {i}, platform: {map_platform_id_to_initial(j)}")
            create_kit_data_from_df(df, j)
