from paje.module.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import gini_index
from paje.util.check import check_float, check_X_y
import pandas as pd


class FilterGiniIndex(Filter):
    """  """

    def __init__(self, ratio=0.8):
        check_float('ratio', ratio, 0.0, 1.0)
        self.ratio = ratio
        self.__rank = self.__score = None

    def fit(self, X, y):
        check_X_y(X, y)

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.__score = gini_index.gini_index(X, y)
        self.__rank = gini_index.feature_ranking(self.__score)
        self.nro_features = int((self.ratio) * X.shape[1])

        return self

    def transform(self, X, y):
        check_X_y(X, y)
        return X[:, self.selected()], y

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def selected(self):
        return self.__rank[0:self.nro_features]
