import numpy as np
from rich.progress import track
from collections import defaultdict

from lori_keystroke_features import (
    all_ids,
    create_kit_data_from_df,
    get_user_by_platform,
    map_platform_id_to_initial,
)
import matplotlib.pyplot as plt


def visualize_feature(feature, original, cleaned, n_bins=15):
    """
    Plot two overlaid histograms (original vs. cleaned) for one feature.
    """
    plt.figure()
    plt.hist(original, bins=n_bins, density=True, alpha=0.5, label="original")
    plt.hist(cleaned, bins=n_bins, density=True, alpha=0.5, label="cleaned")
    plt.title(f"Feature = {feature}")
    plt.legend()
    plt.show()


class HBOS:
    def __init__(self, n_bins=10, contamination=0.1, alpha=0.1, tol=0.5):
        """
        HBOS with smoothing (alpha) and edge‐tolerance (tol).

        Parameters:
        - n_bins:           Base number of bins per feature (can also be 'auto' rules).
        - contamination:    Proportion of expected outliers.
        - alpha:            Small constant added to every bin density.
        - tol:              Fraction of one bin‐width to tolerate just‐outside values.
        """
        self.n_bins = n_bins
        self.contamination = contamination
        self.alpha = alpha
        self.tol = tol

        self.histograms = []  # List of 1D arrays: per-feature bin densities
        self.bin_edges = []  # List of 1D arrays: per-feature edges
        self.feature_names = []  # Keys order

    def fit(self, data: defaultdict):
        """Build (smoothed) histograms for each feature."""
        self.feature_names = list(data.keys())
        X = np.column_stack([data[f] for f in self.feature_names])

        self.histograms.clear()
        self.bin_edges.clear()

        for col in X.T:
            # 1) build raw histogram
            hist, edges = np.histogram(col, bins=self.n_bins, density=True)

            # 2) smooth: add alpha everywhere
            hist = hist + self.alpha

            self.histograms.append(hist)
            self.bin_edges.append(edges)

    def _compute_score(self, x: np.ndarray) -> float:
        """
        Negative‐log‐sum of per‐feature densities with alpha & tol handling.
        Higher score = more anomalous.
        """
        score = 0.0

        for i, xi in enumerate(x):
            edges = self.bin_edges[i]
            hist = self.histograms[i]
            n_bins = hist.shape[0]

            # compute first/last bin widths
            width_low = edges[1] - edges[0]
            width_high = edges[-1] - edges[-2]

            # 1) too far below range?
            if xi < edges[0]:
                if edges[0] - xi <= self.tol * width_low:
                    # snap into first bin
                    density = hist[0]
                else:
                    # true out‐of‐range → worst density
                    density = self.alpha

                score += -np.log(density)
                continue

            # 2) too far above range?
            if xi > edges[-1]:
                if xi - edges[-1] <= self.tol * width_high:
                    # snap into last bin
                    density = hist[-1]
                else:
                    density = self.alpha

                score += -np.log(density)
                continue

            # 3) within [min, max] → find bin index
            bin_idx = np.searchsorted(edges, xi, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            density = hist[bin_idx]
            score += -np.log(density)

        return score

    def decision_function(self, data: defaultdict) -> np.ndarray:
        """Return log‐space HBOS scores for all points."""
        X = np.column_stack([data[f] for f in self.feature_names])
        return np.array([self._compute_score(row) for row in X])

    def clean(self, data: defaultdict) -> defaultdict:
        """
        Remove the top‐contamination fraction of highest‐score points.
        Returns a new defaultdict(list) containing only inliers.
        """
        scores = self.decision_function(data)
        thr = np.percentile(scores, 100 * (1 - self.contamination))
        mask = scores <= thr

        cleaned = defaultdict(list)
        for i, keep in enumerate(mask):
            if keep:
                for f in self.feature_names:
                    cleaned[f].append(data[f][i])

        return cleaned


# Example
if __name__ == "__main__":
    for user_id in track(all_ids(), description="Users"):
        for platform_id in (1, 2, 3):
            df = get_user_by_platform(user_id, platform_id)
            plat_char = map_platform_id_to_initial(platform_id)

            if df.empty:
                print(f"Skipping User: {user_id}, platform: {plat_char}")
                continue

            print(f"\nUser: {user_id}, platform: {plat_char}")

            # 1) extract KIT data per key-pair
            data = create_kit_data_from_df(df, kit_feature_type=platform_id)

            # 2) run HBOS per feature
            cleaned = defaultdict(list)
            for feature, values in data.items():
                single = {feature: values}
                hbos = HBOS(n_bins=15, contamination=0.10, alpha=0.4, tol=0.3)
                hbos.fit(single)
                cleaned[feature] = hbos.clean(single)[feature]

            # → new total‐removed summary:
            total_original = sum(len(v) for v in data.values())
            total_kept = sum(len(v) for v in cleaned.values())
            total_removed = total_original - total_kept

            print(f"Total KIT values:  {total_original}")
            print(f"Kept KIT values:   {total_kept}")
            print(f"Removed outliers:  {total_removed}\n")
