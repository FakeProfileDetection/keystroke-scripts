import os
import math
import statistics
import json
import numpy as np


class StepFunction:
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import (
    >>>     StepFunction)
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    """

    def __init__(self, x, y, ival=0.0, sorted=False, side="left"):  # noqa
        if side.lower() not in ["right", "left"]:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = "x and y must be 1-dimensional"
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """

    def __init__(self, x, side="right"):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1.0 / nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)


class Verify:
    """
    Our implementations of the Similarity (both weighted and unweighted),
    Absolute, Relative, and ITAD verifiers
    """

    def __init__(self, p1, p2, p1_t=10, p2_t=10):
        # p1 and p2 are dictionaries of features
        # keys in the dictionaries would be the feature names
        # feature names mean individual letters for KHT
        # feature names could also mean pair of letters for KIT or diagraphs
        # feature names could also mean pair of sequence of three letters for trigraphs
        # feature names can be extended to any features that we can extract from keystrokes
        self.pattern1 = p1
        self.pattern2 = p2
        self.pattern1threshold = (
            p1_t  # sort of feature selection, based on the availability
        )
        self.pattern2threshold = (
            p2_t  # sort of feature selection, based on the availability
        )
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            config = json.load(f)
        self.common_features = []
        if config["use_feature_selection"]:
            for feature in self.pattern1.keys():
                if feature in self.pattern2.keys():
                    if (
                        len(self.pattern1[feature]) >= self.pattern1threshold
                        and len(self.pattern2[feature]) >= self.pattern2threshold
                    ):
                        self.common_features.append(feature)
        else:
            self.common_features = set(self.pattern1.keys()).intersection(
                set(self.pattern2.keys())
            )
        # print(f"comparing {len(self.common_features)} common_features")

    def get_abs_match_score(self):
        """
        Computes the absolute matching score between two patterns based on their common features.

        The method checks the ratio of medians of each common feature in both patterns. If the ratio is
        below a threshold (currently set to 1.5), it considers the feature as a match. The final score
        is the proportion of matched features to the total common features.

        The function assumes that the class instance has the attributes:
        - self.common_features: a list of features that are common between two patterns.
        - self.pattern1: a dictionary where keys are feature names and values are lists of values for pattern 1.
        - self.pattern2: a dictionary where keys are feature names and values are lists of values for pattern 2.

        Returns:
        - float: The absolute matching score which is a ratio of matched features to total common features.

        Raises:
        - ValueError: If there are no common features or if an unexpected zero median is encountered.

        Notes:
        If there are no common features or minimum of medians of a feature is 0, the function currently returns a score of 0.
        """
        if len(self.common_features) == 0:  # if there are no common features
            return 0

        matches = 0
        for feature in self.common_features:
            raw_pattern1 = self.pattern1[feature]
            raw_pattern2 = self.pattern2[feature]

            # Filter out NaNs to compute means
            valid_p1 = [x for x in raw_pattern1 if not math.isnan(x)]
            valid_p2 = [x for x in raw_pattern2 if not math.isnan(x)]

            if not valid_p1 or not valid_p2:
                print(f"Skipping feature '{feature}' due to all values being NaN.")
                continue

            mean_p1 = statistics.mean(valid_p1)
            mean_p2 = statistics.mean(valid_p2)

            # Replace NaNs with the respective mean
            pattern1_clean = [x if not math.isnan(x) else mean_p1 for x in raw_pattern1]
            pattern2_clean = [x if not math.isnan(x) else mean_p2 for x in raw_pattern2]

            pattern1_median = statistics.median(pattern1_clean)
            pattern2_median = statistics.median(pattern2_clean)

            if min(pattern1_median, pattern2_median) == 0:
                print(f"Skipping feature '{feature}' due to zero median.")
                continue  # Skip the feature if the median is zero
            else:
                ratio = max(pattern1_median, pattern2_median) / min(pattern1_median, pattern2_median)

            threshold = 1.5
            if ratio <= threshold:  # If the feature matches based on median ratio
                matches += 1

        if len(self.common_features) > 0:
            return matches / len(self.common_features)
        else:
            print("No valid matches computed; returning 0.")
            return 0
    def get_similarity_score(self):  # S verifier, each key same weight
        """
        Computes the similarity score between two patterns based on their common features.

        The similarity score is calculated by first computing the median and standard deviation (stdev)
        of the time values for each common feature in pattern 1. For each time value in pattern 2 for the same
        feature, the function checks if the value lies within one standard deviation from the median of pattern 1.

        A feature is considered a match if more than half of its time values in pattern 2 lie within this range.
        The final similarity score is the ratio of matched features to the total common features.

        The function assumes that the class instance has the attributes:
        - self.common_features: a list of features that are common between two patterns.
        - self.pattern1: a dictionary where keys are feature names and values are lists of values for pattern 1.
        - self.pattern2: a dictionary where keys are feature names and values are lists of values for pattern 2.

        Returns:
        - float: The similarity score which is a ratio of matched features to total common features.

        Notes:
        If there are no common features, the function returns a score of 0. In the case where the standard deviation
        cannot be computed (e.g., when a feature has only one value), the function defaults to using a quarter of
        the median value as the stdev.
        """

        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        key_matches, total_features = 0, 0
        for feature in self.common_features:
            pattern1_median = statistics.median(list(self.pattern1[feature]))
            try:
                pattern1_stdev = statistics.stdev(self.pattern1[feature])
            except statistics.StatisticsError:
                # print("In error: ", self.pattern1[feature])
                if len(self.pattern1[feature]) == 1:
                    pattern1_stdev = self.pattern1[feature][0] / 4
                else:
                    pattern1_stdev = (
                        self.pattern1[feature] / 4
                    )  # this will always be one value that is when exception would occur

            value_matches, total_values = 0, 0
            for time in self.pattern2[feature]:
                if (pattern1_median - pattern1_stdev) < time and time < (
                    pattern1_median + pattern1_stdev
                ):
                    value_matches += 1
                total_values += 1
            if value_matches / total_values <= 0.5:
                key_matches += 1
            total_features += 1

        return key_matches / total_features

    def get_weighted_similarity_score(self):
        """
        Computes the weighted similarity score between two patterns based on their common features.

        The weighted similarity score is calculated using the following steps:
        1. Compute the median (as a proxy for the mean) and standard deviation (stdev) of the time values for
        each common feature in pattern 1 (referred to as the "enrollment" pattern).
        2. For each time value in pattern 2 (referred to as the "template" pattern) for the same feature,
        check if the value lies within one standard deviation from the median of the enrollment pattern.
        3. Sum the number of matches and total values for all features.
        4. Compute the ratio of total matches to total values for all features to get the final similarity score.

        Returns:
        - float: The weighted similarity score which is a ratio of total matched values to total values across all
                features.
        """

        if len(self.common_features) == 0:  # if there exist no common features,
            return 0

        matches, total = 0, 0

        for feature in self.common_features:
            # Filter out NaN values from pattern1 and pattern2 for this feature
            valid_p1 = [x for x in self.pattern1[feature] if not math.isnan(x)]
            valid_p2 = [x for x in self.pattern2[feature] if not math.isnan(x)]
            if (
                len(valid_p1) == 0 or len(valid_p2) == 0
            ):  # skip feature if there are no valid values left after removal
                continue
            mean_p1 = statistics.mean(valid_p1)
            mean_p2 = statistics.mean(valid_p2)

            # Replace NaNs with the respective mean
            pattern1_clean = [x if not math.isnan(x) else mean_p1 for x in self.pattern1[feature]]
            pattern2_clean = [x if not math.isnan(x) else mean_p2 for x in self.pattern2[feature]]


            # Compute the median and standard deviation for pattern1
            enroll_mean = statistics.median(pattern1_clean)
            try:
                template_stdev = statistics.stdev(pattern1_clean)
            except statistics.StatisticsError:
                if len(pattern1_clean) == 1:
                    template_stdev = pattern1_clean[0] / 4
                else:
                    template_stdev = pattern1_clean / 4

            # Compare time values from pattern2 against the median ± stdev from pattern1
            for time in pattern2_clean:
                if (
                    (enroll_mean - template_stdev)
                    < time
                    < (enroll_mean + template_stdev)
                ):
                    matches += 1
                total += 1

        # If no valid comparisons were made, return 0 similarity score
        if total == 0:
            return 0

        return matches / total

    def get_cdf_xi(self, distribution, sample):
        """
        Computes the cumulative distribution function (CDF) value at a given sample
        point based on the provided distribution.

        Parameters:
        - distribution (list or array-like): The list of data points representing the distribution.
        - sample (float or int): The point at which to evaluate the CDF.

        Returns:
        - float: The CDF value of the given sample in the provided distribution.
        """
        ecdf = ECDF(distribution)
        prob = ecdf(sample)
        # print('prob:', prob)
        return prob

    def itad_similarity(self):
        """
        Computes the ITAD similarity score
        between two typing patterns based on their shared features.

        The score represents the similarity between two patterns based on the cumulative
        distribution function (CDF) of the median values of the shared features.
        If a value from pattern2 is less than or equal to the median of the corresponding
        feature in pattern1, the CDF value at that point is used. Otherwise, 1 minus the CDF
        value is used.

        Returns:
        - float: The ITAD similarity score, which is the average of the computed similarities
        for all shared features.
        """

        # https://www.scitepress.org/Papers/2023/116841/116841.pdf
        if len(self.common_features) == 0:
            return 0

        similarities = []

        for feature in self.common_features:
            raw_pattern1 = self.pattern1[feature]
            raw_pattern2 = self.pattern2[feature]
            # print(f"Data for pattern1 feature: '{raw_pattern1}'")
            # print(f"Data for pattern2 feature: '{raw_pattern2}'")
            # Filter out NaNs to compute means
            valid_p1 = [x for x in raw_pattern1 if not math.isnan(x)]
            valid_p2 = [x for x in raw_pattern2 if not math.isnan(x)]

            if len(valid_p1) == 0 or len(valid_p2) == 0:
                print(f"Skipping feature '{feature}' due to all values being NaN.")
                continue
            # print(valid_p1)
            # print(valid_p2)
            mean_p1 = statistics.mean(valid_p1)
            mean_p2 = statistics.mean(valid_p2)

            # Replace NaNs with the respective mean
            pattern1_clean = [x if not math.isnan(x) else mean_p1 for x in raw_pattern1]
            pattern2_clean = [x if not math.isnan(x) else mean_p2 for x in raw_pattern2]

            M_g_i = statistics.median(pattern1_clean)
            # print(f"M_g_i (median for feature '{feature}'): {M_g_i}")

            for x_i in pattern2_clean:
                if x_i <= M_g_i:
                    similarities.append(self.get_cdf_xi(pattern1_clean, x_i))
                else:
                    similarities.append(1 - self.get_cdf_xi(pattern1_clean, x_i))

        if similarities:
            return statistics.mean(similarities)
        else:
            print("No valid similarities computed; returning 0.")
            return 0

    def scaled_manhattan_distance(self):
        """
        Computes the Scaled Manhattan Distance between two typing patterns based on their shared features.

        This metric calculates the distance by taking the absolute difference between
        a value from one pattern and the mean of the corresponding feature in the other pattern.
        This difference is then scaled by dividing it with the standard deviation of the
        corresponding feature. The computed distances for all shared features are then averaged
        to provide a final score.

        The Scaled Manhattan Distance gives an insight into how different the two patterns are
        in terms of their common features while accounting for the variability (standard deviation)
        of the features.

        Returns:
        - float: The averaged scaled manhattan distance for all shared features.

        """
        if (
            len(self.common_features) == 0
        ):  # this needs to be checked further when and why and for which users or cases it might hapens at all
            # print("dig deeper: there is no common feature to match!")
            return 0
        grand_sum = 0
        number_of_instances_compared = 0
        for feature in self.common_features:
            # print('comparing the feature:', feature)
            mu_g = statistics.mean(self.pattern1[feature])
            std_g = statistics.stdev(self.pattern1[feature])
            # print(f'mu_g:{mu_g}, and std_g:{std_g}')
            for x_i in self.pattern2[feature]:
                # print('x_i:', x_i)
                current_dist = abs(mu_g - x_i) / std_g
                # print('current_dist:', current_dist)
                grand_sum = grand_sum + current_dist
                # print('grand_sum:', grand_sum)
                number_of_instances_compared = number_of_instances_compared + 1
        # print('number_of_instances_compared', number_of_instances_compared)
        return grand_sum / number_of_instances_compared
