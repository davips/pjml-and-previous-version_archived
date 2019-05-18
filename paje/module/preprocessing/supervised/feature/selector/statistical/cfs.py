from paje.module.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import CFS
from paje.base.hps import HPTree
import pandas as pd


class FilterCFS(Filter):
    def build_impl(self):
        # TODO: forcing to recalculate, since there is no self.model.
        self.memoize = False

        self.__rank = self.__score = self._selected = None

    def apply_impl(self, data):
        X, y = data.xy

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes
        self._selected = CFS.cfs(X, y)
        self._selected = self._selected[self._selected >= 0]

        # self.fit(data.X, data.Y)
        return self.use(data)

    def selected(self):
        return self._selected

    @classmethod
    def tree_impl(cls, data):
        return HPTree(dic={}, children=[])
