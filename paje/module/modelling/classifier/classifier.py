from abc import ABC

import numpy as np

from paje.component import Component


class Classifier(Component, ABC):
    def apply_impl(self, data):
        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.

        self.model.fit(*data.xy())  # self.model will be set in the child class

        if not self.show_warnings:
            np.warnings.filterwarnings('always')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.

    def use_impl(self, data):
        return self.model.predict(data.data_x)
