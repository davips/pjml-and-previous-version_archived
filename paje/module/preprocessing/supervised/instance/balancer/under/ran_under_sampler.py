from imblearn.under_sampling import RandomUnderSampler

from paje.base.hps import HPTree
from paje.module.preprocessing.supervised.instance.balancer.resampler \
    import Resampler


class RanUnderSampler(Resampler):
    def instantiate_impl(self):
        self.model = RandomUnderSampler(**self.dic)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['majority', 'not minority',
                                           'not majority', 'all']]}
        return HPTree(dic, children=[])
