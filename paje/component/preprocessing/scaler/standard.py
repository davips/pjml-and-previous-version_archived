from sklearn.preprocessing import StandardScaler

from paje.base.hps import HPTree
from paje.component.preprocessing.scaler.scaler import Scaler


class Standard(Scaler):
    def __init__(self, **kwargs):
        with_mean, with_std = kwargs.get('with_mean/std')
        del kwargs['with_mean/std']
        self.model = StandardScaler(with_mean, with_std, **kwargs)

    @classmethod
    def hps_impl(cls, data=None):
        dic = {
            'with_mean/std': ['c', [(True, False), (False, True), (True, True)]],  # (False, False) seems to be useless
        }
        return HPTree(dic, children=[])
