from sklearn.naive_bayes import GaussianNB

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class NB(Classifier):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)

        self.model = GaussianNB()

    @classmethod
    def tree_impl(cls, data=None):
        dic = {}
        return HPTree(dic, children=[])
