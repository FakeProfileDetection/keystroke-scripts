import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ExtendedMinMaxScalar(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range=feature_range, copy=copy)

    def fit(self, X, y=None):
        # Call parent fit to initialize internals
        super().fit(X, y)

        # Extend min and max
        range_extension = 0.1 * (self.data_max_ - self.data_min_)
        self.data_min_ = self.data_min_ - range_extension
        self.data_max_ = self.data_max_ + range_extension

        # Recalculate scale_ and min_ for transform
        data_range = self.data_max_ - self.data_min_
        scale = (self.feature_range[1] - self.feature_range[0]) / data_range
        min_ = self.feature_range[0] - self.data_min_ * scale

        self.scale_ = scale
        self.min_ = min_

        return self

    def transform(self, X):
        # Apply the custom transformation based on modified range
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return X_scaled

    def inverse_transform(self, X):
        # Revert the scaling
        X_inv_std = (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        X_inv = X_inv_std * (self.data_max_ - self.data_min_) + self.data_min_
        return X_inv

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
