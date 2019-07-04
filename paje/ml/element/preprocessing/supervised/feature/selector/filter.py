from abc import ABC

from paje.base.component import Component
from paje.base.hps import HPTree
from paje.ml.element.element import Element


class Filter(Element, ABC):
    """ Filter base class"""
    def build_impl(self):
        self.ratio = self.args_set['ratio']
        self._rank = self._score = self._nro_features = None
        self.model = 42 # TODO: better model here?

    def use_impl(self, data):
        return data.updated(self, X=data.X[:, self.selected()])

    def rank(self):
        return self._rank

    def score(self):
        return self._score

    def selected(self):
        return self._rank[0:self._nro_features].copy()

    def tree_impl(cls, data):
        return HPTree(
            # TODO: check if it would be better to adopt a 'z' hyperparameter
            node={'ratio': ['r', [1e-05, 1]]},
            children=[])
