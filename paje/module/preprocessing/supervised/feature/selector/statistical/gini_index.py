import pandas as pd

from paje.module.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import gini_index


class FilterGiniIndex(Filter):
    """  """
    def apply_impl(self, data):
        # TODO: verify if is possible implement this with numpy
        X, y = data.xy()
        y = pd.Categorical(y).codes

        self.__score = gini_index.gini_index(X, y)
        self.__rank = gini_index.feature_ranking(self.__score)
        self.nro_features = int(self.ratio * X.shape[1])

        return self.use(data)