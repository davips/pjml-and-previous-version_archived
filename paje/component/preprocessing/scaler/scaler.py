from abc import ABC

import numpy as np

from paje.component.component import Component


class Scaler(Component, ABC):
    def apply(self, data):
        if not self.__class__.show_warnings:
            np.warnings.filterwarnings('ignore')

        self.model.fit(*data.xy())  # self.model will be set in the child class

        if not self.__class__.show_warnings:
            np.warnings.filterwarnings('always')

    def use(self, data):
        return self.model.transform(data.data_x)
