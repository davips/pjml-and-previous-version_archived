""" Scaler Module
"""
from abc import ABC

from paje.base.component import Component
from paje.base.data import Data


class Scaler(Component, ABC):
    def apply_impl(self, data):
        # self.model will be set in the child class
        self.model.fit(*data.xy)
        return self.use(data)

    def use_impl(self, data):
        return data.updated(X=self.model.transform(data.X))

    def isdeterministic(self):
        return True
