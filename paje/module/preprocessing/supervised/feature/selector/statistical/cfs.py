from paje.module.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import CFS
from paje.base.hps import HPTree
import pandas as pd


class FilterCFS(Filter):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)

        self.__rank = self.__score = self.idx = None

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def apply_impl(self, data):
        X, y = data.xy()

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes
        self.idx = CFS.cfs(X, y)
        self.idx = self.idx[self.idx >= 0]

        # self.fit(data.data_x, data.data_y)
        return self.use(data)

    def use_impl(self, data):
        data.data_x = data.data_x[:, self.idx]
        return data

    @classmethod
    def tree_impl(cls, data=None):
        return HPTree(dic={}, children=[])
