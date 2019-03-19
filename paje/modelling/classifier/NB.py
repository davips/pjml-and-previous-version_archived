from sklearn.naive_bayes import GaussianNB

from paje.base.hps import HPTree


class NB:
    def __init__(self):
        self.model = None

    def apply(self, data):
        X, y = data.xy()
        self.model = GaussianNB().fit(X, y)

    def use(self, data):
        X = data.data_x
        return self.model.predict(X)

    def explain(self, X):
        raise NotImplementedError("Should it return probability distributions?")

    @staticmethod
    def hps(data):
        data = {}
        return HPTree(data, children=[])
