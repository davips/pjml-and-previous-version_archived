from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from paje.base.hps import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class NB(Classifier):
    """NB that accepts any values."""

    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.

        newconfig = self.config.copy()
        self.nb_type = newconfig.get('@nb_type')
        del newconfig['@nb_type']

        if self.nb_type == "GaussianNB":
            self.model = GaussianNB()
        elif self.nb_type == "BernoulliNB":
            self.model = BernoulliNB()
        else:
            raise Exception('Wrong NB!')

    @classmethod
    def tree_impl(self):
        node = {'@nb_type': ['c', ["GaussianNB", "BernoulliNB"]]}
        return HPTree(node=node, children=[])
