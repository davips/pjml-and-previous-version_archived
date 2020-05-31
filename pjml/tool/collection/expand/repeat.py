from itertools import repeat

from pjdata.collection import Collection
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.nonfinalizer import NonFinalizer
from pjml.tool.abc.mixin.component import Component


class Repeat(NonFinalizer, Component):
    """Data -> Collection"""

    @property
    def finite(self):
        return False

    def enhancer_info(self):
        return {}

    def model_info(self, data):
        return {}

    def enhancer_func(self):
        return lambda d: Collection(repeat(d), lambda: d, finite=False,
                                    debug_info='expand')

    def model_func(self, data):
        return self.enhancer_func()

    # def iterators(self, train, test):
    #     return repeat(train), repeat(test)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
