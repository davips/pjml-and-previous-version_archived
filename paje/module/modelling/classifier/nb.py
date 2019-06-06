from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class NB(Classifier):
    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.
        newdic = self.dic.copy()
        self.nb_type = newdic.get('@nb_type')
        del newdic['@nb_type']

        if self.nb_type == "GaussianNB":
            self.model = GaussianNB()
        elif self.nb_type == "MultinomialNB":
            self.model = MultinomialNB()
            self.model = GaussianNB()
        elif self.nb_type == "ComplementNB":
            self.model = ComplementNB()
            self.model = GaussianNB()
        elif self.nb_type == "BernoulliNB":
            self.model = BernoulliNB()

    def apply_impl(self, data):
        X, y = data.Xy
        if self.nb_type in ["MultinomialNB", "ComplementNB"]:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        # self.model will be set in the child class
        self.model.fit(X, y)
        return self.use_impl(data)

    def use_impl(self, data):
        X, y = data.Xy
        if self.nb_type in ["MultinomialNB", "ComplementNB"]:
            X = self.scaler.transform(X)

        return data.updated(z=self.model.predict(X))


    @classmethod
    def tree_impl(cls, data=None):
        return HPTree(dic={'@nb_type': ['c', ["GaussianNB", "MultinomialNB",
                                    "ComplementNB", "BernoulliNB"]]},
                      children=[])
