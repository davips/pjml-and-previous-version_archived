from sklearn.preprocessing import MinMaxScaler

from paje.base.hps import HPTree
from paje.ml.element.preprocessing.unsupervised.feature.scaler.scaler import Scaler


class Equalization(Scaler):
    def build_impl(self, **config):
        newconfig = self.config.copy()
        self.model = MinMaxScaler(**newconfig)

    @classmethod
    def tree_impl(cls, data=None):
        node = {
            'feature_range': ['c', [(-1, 1), (0, 1)]],
        }
        return HPTree(node, children=[])
