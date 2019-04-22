from paje.module.preprocessing.feature_selection.filter import Filter
from skfeature.function.statistical_based import chi_square
from paje.util.check import check_float
from paje.base.hps import HPTree
import pandas as pd
import math


class FilterChiSquare(Filter):
    """  """

    def init_impl(self, ratio=0.8):
        check_float('ratio', ratio, 0.0, 1.0)
        self.ratio = ratio
        self.rank = self.score = None
        self.nro_features = None

    def rank(self):
        return self.rank.copy()

    def score(self):
        return self.score.copy()

    def selected(self):
        return self.rank[0:self.nro_features]

    def apply_impl(self, data):
        X, y = data.xy()
        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.score = chi_square.chi_square(X, y)  # Input X must be non-negative. <- This happens when some scaler generates negative values.
        self.rank = chi_square.feature_ranking(self.score)
        self.nro_features = math.ceil((self.ratio) * X.shape[1])

        return self.use(data)

    def use_impl(self, data):
        data.data_x = data.data_x[:, self.selected()]
        return data

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        return HPTree(
            dic={'ratio': ['r', 1e-05, 1]},
            children=[])
