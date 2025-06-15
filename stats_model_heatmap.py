import os
import tqdm
import json
import enum
import matplotlib.pyplot as plt
import seaborn as sns
from lori_keystroke_features import (
    all_ids,
    create_kht_data_from_df,
    create_kit_data_from_df,
    get_user_by_platform,
)
from rich.progress import track
from verifiers_core import Verify


class VerifierType(enum.Enum):
    """Enum class representing the different types of verifiers available."""

    RELATIVE = 1
    SIMILARITY = 2
    SIMILARITY_UNWEIGHTED = 3
    ABSOLUTE = 4
    ITAD = 5


class HeatMap:
    """
    A heatmap generates the representative matrices for KHT, KIT, optionally word level features, or all of them
    and making a heatmap plot out of it
    """

    def __init__(self, verifier_type, p1=10, p2=10):
        self.verifier_type = verifier_type  # The verifier class to be used
        self.p1_threshold = p1
        self.p2_threshold = p2
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            self.config = json.load(f)
        print(f"----selected {verifier_type}")

    def make_kht_matrix(
        self, enroll_platform_id, probe_platform_id, enroll_session_id, probe_session_id
    ):
        """
        Make a matrix of KHT features from the enrollment and probe id's and
        their respective session id's
        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """
        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")

        matrix = []
        ids = all_ids()
        for i in track(ids):
            print(i)
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kht_data_from_df(df)
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kht_data_from_df(df)
                v = Verify(enrollment, probe, self.p1_threshold, self.p2_threshold)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def make_kit_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        """
        Make a matrix of KIT features from the enrollment and probe id's,
        their respective session id's, and the KIT flight feature (1-4)

        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """

        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        print(self.verifier_type)
        matrix = []
        ids = all_ids()
        for i in track(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kit_data_from_df(df, kit_feature_type)
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kit_data_from_df(df, kit_feature_type)
                v = Verify(enrollment, probe)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def combined_keystroke_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        """
        Make a combined matrix of KIT and KHT features

        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """
        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")

        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        ids = all_ids()
        for i in tqdm.tqdm(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            print(
                f"enroll_platform_id: {enroll_platform_id}, enroll_session_id: {enroll_session_id}, df.shape: {df.shape}"
            )
            print(
                f"probe_platform_id: {probe_platform_id}, probe_session_id: {probe_session_id}, df.shape: {df.shape}"
            )
            kht_enrollment = create_kht_data_from_df(df)
            kit_enrollment = create_kit_data_from_df(df, kit_feature_type)
            combined_enrollment = kht_enrollment | kit_enrollment
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                kht_probe = create_kht_data_from_df(df)
                kit_probe = create_kit_data_from_df(df, kit_feature_type)
                combined_probe = kht_probe | kit_probe
                v = Verify(combined_enrollment, combined_probe)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def plot_heatmap(self, matrix, title=None):
        """Generate a heatmap from the provided feature matrix and optional title"""
        ax = sns.heatmap(matrix, linewidth=0.5).set_title(title)
        plt.savefig(title)
