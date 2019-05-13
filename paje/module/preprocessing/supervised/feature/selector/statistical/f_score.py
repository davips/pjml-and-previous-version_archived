from paje.base.hps import HPTree
from paje.module.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import f_score
from paje.util.check import check_float
import pandas as pd
import math


class FilterFScore(Filter):
    """  """
    def apply_impl(self, data):
        X, y = data.xy()

        # TODO: verify if it is possible implement this with numpy
        y = pd.Categorical(y).codes

        self._score = f_score.f_score(X, y)
        self._rank = f_score.feature_ranking(self._score)
        self._nro_features = math.ceil(self.ratio * X.shape[1])

        return self.use(data)
