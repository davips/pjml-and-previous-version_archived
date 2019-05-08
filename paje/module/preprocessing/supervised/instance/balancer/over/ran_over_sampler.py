from imblearn.over_sampling import RandomOverSampler

from paje.base.hps import HPTree
from paje.module.preprocessing.supervised.instance.balancer.resampler import Resampler


class RanOverSampler(Resampler):
    def init_impl(self, **kwargs):
        self.model = RandomOverSampler(**kwargs)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['not minority', 'not majority', 'all']]}
        return HPTree(dic, children=[])
