from imblearn.under_sampling import RandomUnderSampler

from paje.base.hps import HPTree
from paje.module.preprocessing.balancer.resampler import Resampler


class RanUnderSampler(Resampler):
    def init_impl(self, **kwargs):
        self.model = RandomUnderSampler(**kwargs)

    @classmethod
    def hps_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['majority', 'not minority', 'not majority', 'all']]}
        return HPTree(dic, children=[])
