""" Scaler Module
"""
from abc import ABC

from paje.base.component import Component
from paje.base.data import Data


class Scaler(Component, ABC):
    # def touched_fields(self):
    #     return 'X'
    #
    # def still_compatible_fields(self):
    #     return 'all'
    #
    # def needed_fields(self):
    #     return 'X'

    def apply_impl(self, data):
        # self.model will be set in the child class
        self.model.fit(*data.Xy)
        return self.use_impl(data)

    def use_impl(self, data):
        return data.updated(self, X=self.model.transform(data.X))

    def isdeterministic(self):
        return True
