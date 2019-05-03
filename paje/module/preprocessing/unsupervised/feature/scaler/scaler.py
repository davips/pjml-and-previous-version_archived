from abc import ABC

import numpy as np

from paje.base.component import Component


class Scaler(Component, ABC):
    def apply_impl(self, data):
        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')

        self.model.fit(*data.xy())  # self.model will be set in the child class

        if not self.show_warnings:
            np.warnings.filterwarnings('always')

        return self.use(data)

    def use_impl(self, data):
        data.data_x = self.model.transform(data.data_x)
        return data
