from paje.module.preprocessing.feature_selection import Filter
from skfeature.function.statistical_based import t_score
from paje.util.check import check_float, check_X_y
import pandas as pd
import numpy as np
from itertools import combinations


class FilterTScore(Filter):
    """  """
    def __init__(self, ratio=0.8):
        check_float('ratio', ratio, 0.0, 1.0)
        self.ratio = ratio
        self.__rank = self.__score = None


    def comb_idx(self, n,k):
        return np.array(list(combinations(range(n), k)))


    def apply_t_score(self, X, y):
        cat = np.unique(y)
        cat_len = len(cat)
        idx_cat = [y == i for i in cat]
        aux = []
        rank_point = np.zeros(X.shape[1])

        for a, b in self.comb_idx(cat_len, 2):
            idx = np.logical_or(idx_cat[a], idx_cat[b])
            score = t_score.t_score(X[idx], y[idx])
            aux.append(score)

        # If there is more one class we compute the rank by the average
        # of the score
        self.__score = np.sum(aux, axis=0)
        self.__rank = np.argsort(self.__score)[::-1]


    def fit(self, X, y):
        check_X_y(X, y)

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.apply_t_score(X, y)
        self.nro_features = int((self.ratio)*X.shape[1])

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
