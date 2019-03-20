from sklearn.naive_bayes import GaussianNB

from paje.base.hps import HPTree
from paje.modelling.classifier.classifier import Classifier


class NB(Classifier):
    def __init__(self):
        self.model = GaussianNB()

    @staticmethod
    def hps_impl(data):
        dic = {}
        return HPTree(dic, children=[])
