from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.module.preprocessing.scaler.scaler import Scaler


class Equalization(Scaler):
    def init_impl(self, **kwargs):
        self.model = MinMaxScaler(**kwargs)

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        dic = {
            'feature_range': ['c', [(-1, 1), (0, 1)]],
        }
        return HPTree(dic, children=[])
