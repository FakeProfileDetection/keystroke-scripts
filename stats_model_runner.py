import numpy as np
from sklearn.metrics import top_k_accuracy_score

from lori_keystroke_features import all_ids
from stats_model_heatmap import HeatMap, VerifierType
from tabulate import tabulate


def print_k_table(matrix, ids):
    rows = []
    rows.append([1, top_k_accuracy_score(np.array(ids), np.array(matrix), k=1)])
    rows.append([2, top_k_accuracy_score(np.array(ids), np.array(matrix), k=2)])
    rows.append([3, top_k_accuracy_score(np.array(ids), np.array(matrix), k=3)])
    rows.append([4, top_k_accuracy_score(np.array(ids), np.array(matrix), k=4)])
    rows.append([5, top_k_accuracy_score(np.array(ids), np.array(matrix), k=5)])
    table = tabulate(rows, headers=["K", "Score"], tablefmt="plain")
    print(table)


def same_platform_even_split():
    heatmap = HeatMap(VerifierType.SIMILARITY)

    matrix = heatmap.combined_keystroke_matrix(1, 1, [1, 3], [4, 6], 1)
    matrix2 = heatmap.combined_keystroke_matrix(2, 2, [1, 3], [4, 6], 1)
    matrix3 = heatmap.combined_keystroke_matrix(3, 3, [1, 3], [4, 6], 1)

    ids = all_ids()
    print()
    print("Facebook")
    print_k_table(matrix=matrix, ids=ids)
    print("Instagram")
    print_k_table(matrix=matrix2, ids=ids)
    print("Twitter")
    print_k_table(matrix=matrix3, ids=ids)


def train_session_one_test_two():
    heatmap = HeatMap(VerifierType.ITAD)

    matrix = heatmap.combined_keystroke_matrix(1, 1, 1, 4, 1)
    matrix2 = heatmap.combined_keystroke_matrix(2, 2, 1, 4, 1)
    matrix3 = heatmap.combined_keystroke_matrix(3, 3, 1, 4, 1)
    ids = all_ids()
    print()
    print("Facebook")
    print_k_table(matrix=matrix, ids=ids)
    print("Instagram")
    print_k_table(matrix=matrix2, ids=ids)
    print("Twitter")
    print_k_table(matrix=matrix3, ids=ids)


def train_on_one_test_another(verifier_type=VerifierType):
    heatmap = HeatMap(verifier_type)

    matrix = heatmap.combined_keystroke_matrix(1, 2, None, None, 1)
    matrix2 = heatmap.combined_keystroke_matrix(1, 3, None, None, 1)
    matrix3 = heatmap.combined_keystroke_matrix(2, 1, None, None, 1)
    matrix4 = heatmap.combined_keystroke_matrix(2, 3, None, None, 1)
    matrix5 = heatmap.combined_keystroke_matrix(3, 1, None, None, 1)
    matrix6 = heatmap.combined_keystroke_matrix(3, 2, None, None, 1)
    ids = all_ids()
    print()
    print("F vs. I")
    print_k_table(matrix=matrix, ids=ids)
    print("F vs. T")
    print_k_table(matrix=matrix2, ids=ids)
    print("I vs. F")
    print_k_table(matrix=matrix3, ids=ids)
    print("I vs. T")
    print_k_table(matrix=matrix4, ids=ids)
    print("T vs. F")
    print_k_table(matrix=matrix5, ids=ids)
    print("T vs. I")
    print_k_table(matrix=matrix6, ids=ids)


def cross_platform_2v1(verifier_type=VerifierType):
    heatmap = HeatMap(verifier_type)
    matrix = heatmap.make_kit_matrix([1, 2], 3, None, None, 1)
    matrix2 = heatmap.make_kit_matrix([1, 3], 2, None, None, 1)
    matrix3 = heatmap.make_kit_matrix([2, 1], 3, None, None, 1)
    matrix4 = heatmap.make_kit_matrix([2, 3], 1, None, None, 1)
    matrix5 = heatmap.make_kit_matrix([3, 1], 2, None, None, 1)
    matrix6 = heatmap.make_kit_matrix([3, 2], 1, None, None, 1)
    ids = all_ids()
    print()
    print("FI")
    print_k_table(matrix=matrix, ids=ids)

    print("FT")
    print_k_table(matrix=matrix2, ids=ids)

    print("IF")
    print_k_table(matrix=matrix3, ids=ids)

    print("IT")
    print_k_table(matrix=matrix4, ids=ids)

    print("TF")
    print_k_table(matrix=matrix5, ids=ids)

    print("TI")
    print_k_table(matrix=matrix6, ids=ids)


if __name__ == "__main__":
    # train_on_one_test_another(VerifierType.ABSOLUTE)
    # train_on_one_test_another(VerifierType.SIMILARITY)
    # train_on_one_test_another(VerifierType.ITAD)
    # cross_platform_2v1(VerifierType.ABSOLUTE)
    # cross_platform_2v1(VerifierType.SIMILARITY)
    cross_platform_2v1(VerifierType.ITAD)
