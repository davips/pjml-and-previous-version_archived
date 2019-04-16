from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.component.preprocessing.scaler.scaler import Scaler


class Equalization(Scaler):
    def __init__(self, **kwargs):
        self.model = MinMaxScaler(**kwargs)

    @classmethod
    def hps_impl(cls, data=None):
        dic = {
            'feature_range': ['c', [(-1, 1), (0, 1)]],
        }
        return HPTree(dic, children=[])
