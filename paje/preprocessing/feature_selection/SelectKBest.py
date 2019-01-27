from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import low_variance
from skfeature.function.statistical_based import t_score
import pandas as pd
import numpy as np
from itertools import combinations


class Util():

    @staticmethod
    def comb_idx(n,k):
        return np.array(list(combinations(range(n), k)))


class Filter():
    @staticmethod
    def apply_cfs(X, y):
        idx = CFS.cfs(X, y)
        return idx

    @staticmethod
    def apply_chi_square(X, y):
        score = chi_square.chi_square(X, y)
        idx = chi_square.feature_ranking(score)
        return idx

    @staticmethod
    def apply_f_score(X, y):
        score = f_score.f_score(X, y)
        idx = f_score.feature_ranking(score)
        return idx

    @staticmethod
    def apply_gini_index(X, y):
        score = gini_index.gini_index(X, y)
        idx = gini_index.feature_ranking(score)
        return idx


    @staticmethod
    def apply_binary_t_score(X, y):
        score = t_score.t_score(X, y)
        idx = t_score.feature_ranking(score)

        return idx


    @staticmethod
    def apply_t_score(X, y):
        cat = np.unique(y)
        cat_len = len(cat)
        idx_cat = [y == i for i in cat]
        ranks = []
        rank_point = np.zeros(X.shape[1])
        # print(rank_point)

        for a,b in Util.comb_idx(cat_len, 2):
            idx = np.logical_or(idx_cat[a], idx_cat[b])
            ranks.append(Filter.apply_binary_t_score(X[idx], y[idx]))

        # print(ranks)
        # TODO: Improve it
        for r in ranks:
            count = 0
            for i in r:
                # print(i)
                rank_point[i] += count
                count += 1
        # score = t_score.t_score(X, y)
        # idx = t_score.feature_ranking(score)
        # print(rank_point)
        return np.argsort(rank_point)


class SelectKBest():
    """ """
    __methods = {
        "cfs":Filter.apply_cfs,
        "chi_square":Filter.apply_chi_square,
        "f_score":Filter.apply_f_score,
        "gini_index":Filter.apply_gini_index,
        # "low_variance":SelectKBest.__apply_low_variance,
        "t_score":Filter.apply_t_score

    }


    def __init__(self, method="cfs", k=10):
        self.method = SelectKBest.__methods[method]
        self.k = k


    def fit(self, X, y):
        # tranform y to categorical
        X = X.to_numpy()
        y = pd.Categorical(y).codes

        self.idx = self.method(X, y)


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)


    def transform(self, X):
        return X.iloc[:, self.idx[0:self.k]]


    def get_methods():
        return list(SelectKBest.__methods.keys())


