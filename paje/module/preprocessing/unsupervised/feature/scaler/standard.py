from sklearn.preprocessing import StandardScaler

from paje.base.hps import HPTree
from paje.module.preprocessing.unsupervised.feature.scaler.scaler import Scaler


class Standard(Scaler):
    def instantiate_impl(self):
        newdic = self.dic.copy()
        mean_std = newdic.get('@with_mean/std')
        if mean_std is None:
            with_mean, with_std = True, True
        else:
            del newdic['@with_mean/std']
            with_mean, with_std = mean_std
        self.model = StandardScaler(with_mean, with_std, **newdic)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {
            '@with_mean/std':
                ['c', [(True, False), (False, True), (True, True)]]
            # (False, False) seems to be useless
        }
        return HPTree(dic, children=[])
