from imblearn.under_sampling import RandomUnderSampler

from paje.base.hps import HPTree
from paje.ml.element.preprocessing.supervised.instance.sampler.resampler \
    import Resampler


class RanUnderSampler(Resampler):
    def build_impl(self):
        self.model = RandomUnderSampler(**self.args_set)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['majority', 'not minority',
                                           'not majority', 'all']]}
        return HPTree(dic, children=[])
