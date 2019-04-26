from abc import ABC, abstractmethod

from paje.base.component import Component
from paje.base.hps import HPTree


class Reductor(Component, ABC):
    def apply_impl(self, data):
        self.att_labels = data.columns
        max_components = min(data.n_instances(), data.n_attributes())
        if self.model.n_components > max_components:
            self.warning(self.__class__.__name__ + '. Too many components to select from too few attributes or instances!')
            self.model.n_components = max_components
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
