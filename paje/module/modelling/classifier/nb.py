from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class NB(Classifier):
    """NB that accepts any values."""

    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.

        newdic = self.dic.copy()
        self.nb_type = newdic.get('@nb_type')
        del newdic['@nb_type']

        if self.nb_type == "GaussianNB":
            self.model = GaussianNB()
        elif self.nb_type == "BernoulliNB":
            self.model = BernoulliNB()
        else:
            raise Exception('Wrong NB!')

    @classmethod
    def tree_impl(self):
        dic = {'@nb_type': ['c', ["GaussianNB", "BernoulliNB"]]}
        return HPTree(dic=dic, children=[])
