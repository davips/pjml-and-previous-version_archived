
from paje.preprocessing.feature_selection.filter import Filter
from skfeature.function.statistical_based import chi_square
from paje.util.check import check_float, check_X_y
import pandas as pd
from paje.opt.hps import HPTree


class FilterChiSquare(Filter):
    """  """
    def __init__(self, ratio=0.8):
        check_float('ratio', ratio, 0.0, 1.0)
        self.ratio = ratio
        self.__rank = self.__score = None


    def fit(self, X, y):
        check_X_y(X, y)

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.__score = chi_square.chi_square(X, y)
        self.__rank = chi_square.feature_ranking(self.__score)
        self.nro_features = int((self.ratio)*X.shape[1])

        return self

    def transform(self, X):
        # check_X_y(X)
        return X[:, self.selected()]

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def selected(self):
        return self.__rank[0:self.nro_features]

    def hps(self, data):
        return HPTree(
            data={'ratio': ['c', 1e-05, 1]},
            children=None)

    def apply(self, data):
        self.fit(data.data_x, data.data_y)
        data.data_x = self.transform(data.data_x)
        return data

    def use(self, data):
        data.data_x = self.transform(data.data_x)
        return data
