from abc import ABC, abstractmethod

from paje.base.component import Component
from paje.base.hps import HPTree
from paje.data.data import Data


class Reductor(Component, ABC):
    def apply_impl(self, data):
        self.att_labels = data.columns
        self.model.fit(data.data_x)
        data.data_x = self.model.transform(data.data_x)
        return data

    def use_impl(self, data):
        data.data_x = self.model.transform(data.data_x)
        return data

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        cls.check_data(data)
        # TODO: set random_state
        dic = {'n_components': ['z', [1, data.n_attributes()]]}
        dic.update(cls.specific_dictionary(data))
        return HPTree(dic, children=[])

    @classmethod
    @abstractmethod
    def specific_dictionary(cls, data):
        pass
