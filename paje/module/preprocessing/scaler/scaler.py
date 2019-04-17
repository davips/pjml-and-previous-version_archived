from abc import ABC

import numpy as np

from paje.component import Component


class Scaler(Component, ABC):
    def apply_impl(self, data):
        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')

        self.model.fit(*data.xy())  # self.model will be set in the child class

        if not self.show_warnings:
            np.warnings.filterwarnings('always')

    def use_impl(self, data):
        return self.model.transform(data.data_x)
