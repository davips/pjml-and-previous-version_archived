from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.module.preprocessing.unsupervised.feature.scaler.scaler import Scaler


class Equalization(Scaler):
    def instantiate_impl(self):
        del self.dic['random_state']
        newdic = self.dic.copy()
        self.model = MinMaxScaler(**newdic)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {
            'feature_range': ['c', [(-1, 1), (0, 1)]],
        }
        return HPTree(dic, children=[])
