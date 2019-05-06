from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.module.preprocessing.unsupervised.feature.scaler.scaler import Scaler


class Equalization(Scaler):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)
        self.model = MinMaxScaler(**kwargs)

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        dic = {
            'feature_range': ['c', [(-1, 1), (0, 1)]],
        }
        return HPTree(dic, children=[])
