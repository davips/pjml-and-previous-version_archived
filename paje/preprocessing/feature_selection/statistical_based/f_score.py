from paje.preprocessing.feature_selection.filter import Filter
from skfeature.function.statistical_based import f_score
from paje.util.check import check_float, check_X_y
import pandas as pd


class FilterFScore(Filter):
    """  """

    def __init__(self, ratio=0.8):
        check_float('ratio', ratio, 0.0, 1.0)
        self.ratio = ratio
        self.__rank = self.__score = None

    def apply(self, data):
        X, y = data.xy()

        # TODO: verify if it is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.__score = f_score.f_score(X, y)
        self.__rank = f_score.feature_ranking(self.__score)
        self.nro_features = int((self.ratio) * X.shape[1])

        return self

    def use(self, data):
        X, y = data.xy()
        return X[:, self.selected()], y

    def rank(self):
        return self.__rank

    def score(self):
        return self.__score

    def selected(self):
        return self.__rank[0:self.nro_features]
