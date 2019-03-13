from sklearn.ensemble import RandomForestClassifier
from paje.opt.hps import HPTree


class RandomForest():
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.model = None

    def apply(self, data):
        X = data.data_x
        y = data.data_y
        self.fit(X, y)

    def use(self, data):
        X = data.data_x
        return self.model.predict(X)

    def fit(self, X, y=None):
        """Todo the doc string
        """
        self.model = RandomForestClassifier(self.n_estimators,
                                            max_depth=2).fit(X, y)

    def predict(self, X, y=None):
        """Todo the doc string
        """
        return self.model.predict(X)

    @staticmethod
    def hps(data):
        return HPTree(data={'n_estimators': ['z', 2, 1000]},
                      children=[])
