from imblearn.over_sampling import RandomOverSampler

from paje.base.hps import HPTree
from paje.ml.element.preprocessing.supervised.instance.sampler.resampler import Resampler


class RanOverSampler(Resampler):
    def build_impl(self):
        self.model = RandomOverSampler(**self.args_set)

    @classmethod
    def tree_impl(cls, data=None):
        dic = {'sampling_strategy': ['c', ['not minority', 'not majority', 'all']]}
        return HPTree(dic, children=[])
