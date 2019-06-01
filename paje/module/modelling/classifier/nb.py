from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class NB(Classifier):
    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.
        newdic = self.dic.copy()
        self.n_instances = newdic.get('@nb_type')
        del newdic['@nb_type']

        if nb_type == "GaussianNB":
            self.model = GaussianNB()
        elif nb_type == "MultinomialNB":
            self.model = MultinomialNB()
        elif nb_type == "ComplementNB":
            self.model = ComplementNB()
        elif nb_type == "BernoulliNB":
            self.model = BernoulliNB()

    @classmethod
    def tree_impl(cls, data=None):
        return = HPTree(
            dic={'@nb_type': ['c', ["GaussianNB", "MultinomialNB",
                                    "ComplementNB", "BernoulliNB"]])
