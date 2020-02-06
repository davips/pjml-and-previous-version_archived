from imblearn.under_sampling import RandomUnderSampler

from pjml.config.description.cs.componentcs import ComponentCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.data.processing.instance.sampler.resampler import Resampler


class UnderS(Resampler):
    def __init__(self, **kwargs):
        super().__init__(kwargs, RandomUnderSampler(**kwargs))

    @classmethod
    def _cs_impl(cls, data=None):
        params = {
            'sampling_strategy':
                CatP(choice, items=['not minority', 'not majority', 'all'])
        }
        return ComponentCS(Node(params=params))
