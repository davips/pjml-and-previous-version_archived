""" Scaler Module
"""
from abc import ABC

from paje.base.component import Component


class Scaler(Component, ABC):
    def apply_impl(self, data):
        # self.model will be set in the child class
        self.model.fit(*data.xy())
        return self.use(data)

    def use_impl(self, data):
        data.data_x = self.model.transform(data.data_x)
        return data
