from sklearn.naive_bayes import GaussianNB

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class NB(Classifier):
    def instantiate_impl(self):
        self.model = GaussianNB()

    @classmethod
    def tree_impl(cls, data=None):
        dic = {}
        return HPTree(dic, children=[])
