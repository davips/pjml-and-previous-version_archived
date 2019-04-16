from paje.component.preprocessing.feature_selection.filter import Filter
from skfeature.function.statistical_based import CFS
from paje.base.hps import HPTree
import pandas as pd


class FilterCFS(Filter):
    def __init__(self):
        self.__rank = self.__score = self.idx = None

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def apply(self, data):
        X, y = data.xy()

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes
        self.idx = CFS.cfs(X, y)
        self.idx = self.idx[self.idx >= 0]

        # self.fit(data.data_x, data.data_y)
        self.use(data)

    def use(self, data):
        data.data_x = data.data_x[:, self.idx]

    @staticmethod
    def hps_impl(data=None):
        return HPTree(dic={}, children=[])
