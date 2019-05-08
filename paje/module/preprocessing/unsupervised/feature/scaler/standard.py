from sklearn.preprocessing import StandardScaler

from paje.base.hps import HPTree
from paje.module.preprocessing.unsupervised.feature.scaler.scaler import Scaler


class Standard(Scaler):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)
        mean_std = kwargs.get('@with_mean/std')
        if mean_std is None:
            with_mean, with_std = True, True
        else:
            del kwargs['@with_mean/std']
            with_mean, with_std = mean_std
        self.model = StandardScaler(with_mean, with_std, **kwargs)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {
            '@with_mean/std': ['c', [(True, False), (False, True), (True, True)]]  # (False, False) seems to be useless
        }
        return HPTree(dic, children=[])

    def isdeterministic(self):
        return True