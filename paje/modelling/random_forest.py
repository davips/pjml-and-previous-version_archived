from sklearn.ensemble import RandomForestClassifier
from paje.base.hps import HPTree
from paje.base.component import Component


class RandomForest(Component):
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.model = None

    def apply(self, data):
        data_x, data_y = data.xy()
        self.model = RandomForestClassifier(self.n_estimators,
                                            max_depth=2).fit(data_x, data_y)

    def use(self, data):
        data_x = data.data_x
        return self.model.predict(data_x)

    @staticmethod
    def hps(data=None):
        return HPTree(data={'n_estimators': ['z', 2, 1000]},
                      children=[])
