from imblearn.under_sampling import RandomUnderSampler

from paje.base.hps import HPTree
from paje.module.preprocessing.supervised.instance.balancer.resampler import Resampler


class RanUnderSampler(Resampler):
    def init_impl(self, **kwargs):
        self.model = RandomUnderSampler(**kwargs)

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['majority', 'not minority', 'not majority', 'all']]}
        return HPTree(dic, children=[])
