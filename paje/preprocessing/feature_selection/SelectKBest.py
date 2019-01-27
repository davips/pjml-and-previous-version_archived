from external_library.skfeature.function.function.statistical_based import CFS
from external_library.skfeature.function.function.statistical_based import chi_square
from external_library.skfeature.function.function.statistical_based import f_score
from external_library.skfeature.function.function.statistical_based import gini_index
from external_library.skfeature.function.function.statistical_based import low_variance
from external_library.skfeature.function.function.statistical_based import t_socre

class SelectKBest():
    """ """
    __methods = {
        "cfs":SelectKBest.__apply_cfs,
        "chi_square":SelectKBest.__apply_chi_square,
        "f_score":SelectKBest.__apply_f_score,
        "gini_index":SelectKBest.__apply_gini_index,
        "low_variance":SelectKBest.__apply_low_variance,
        "t_score":SelectKBest.__apply_t_score

    }
    def __init__(method="cfs", k=10):
        pass

    def fit(X, y):
        pass

    def show_methods():
        pass

    def __apply_cfs(X, y):
        pass

    def __apply_chi_square(X, y):
        pass

    def __apply_f_score(X, y):
        pass

    def __apply_gini_index(X, y):
        pass

    def __apply_low_variance(X, y):
        pass

    def __apply_t_score(X, y):
        pass

