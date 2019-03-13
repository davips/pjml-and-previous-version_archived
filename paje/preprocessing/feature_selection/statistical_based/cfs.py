from paje.preprocessing.feature_selection.filter import Filter
from skfeature.function.statistical_based import CFS
from paje.util.check import check_float, check_X_y
from paje.opt.hps import HPTree
import pandas as pd


class FilterCFS(Filter):
    def __init__(self):
        self.__rank = self.__score = None

    def fit(self, X, y):
        check_X_y(X, y)

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.idx = CFS.cfs(X, y)
        self.idx = self.idx[self.idx >= 0]

        return self

    def transform(self, X):
        return X[:, self.idx]

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def selected(self):
        return self.idx

    @staticmethod
    def hps(data):
        return HPTree(data=None, children=None)

    def apply(self, data):
        self.fit(data.data_x, data.data_y)
        data.data_x = self.transform(data.data_x)
        return data

    def use(self, data):
        data.data_x = self.transform(data.data_x)
        return data
