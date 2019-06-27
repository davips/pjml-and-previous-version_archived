from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from paje.base.hps import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class NBP(Classifier):
    """NB that needs positive values."""

    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.

        newdic = self.dic.copy()
        self.nb_type = newdic.get('@nb_type')
        del newdic['@nb_type']

        if self.nb_type == "MultinomialNB":
            self.model = MultinomialNB()
        elif self.nb_type == "ComplementNB":
            self.model = ComplementNB()
        else:
            raise Exception('Wrong NB!')

    @classmethod
    def tree_impl(self):
        dic = {'@nb_type': ['c', ["MultinomialNB", "ComplementNB"]]}
        return HPTree(dic=dic, children=[])
