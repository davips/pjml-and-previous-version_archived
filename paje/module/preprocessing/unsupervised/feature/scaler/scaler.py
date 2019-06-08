""" Scaler Module
"""
from abc import ABC

from paje.base.component import Component
from paje.base.data import Data


class Scaler(Component, ABC):
    def fields_to_store_after_use(self):
        return 'X'

    def fields_to_keep_after_use(self):
        return 'y'

    def apply_impl(self, data):
        # self.model will be set in the child class
        self.model.fit(*data.Xy)
        return self.use_impl(data)

    def use_impl(self, data):
        return data.updated(self, X=self.model.transform(data.X))

    def isdeterministic(self):
        return True
