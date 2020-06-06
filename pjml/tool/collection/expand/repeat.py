from typing import Tuple, Iterator

from itertools import repeat

from pjdata.content.collection import Collection
from pjdata.content.data import Data
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.nonfinalizer import NonFinalizer
from pjml.tool.abc.mixin.component import Component


class Repeat(NonFinalizer, Component):
    """Data -> Collection"""

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

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

    def iterator(self, train: Data, test: Data) -> Iterator[Tuple[Data, Data]]:
        return zip(repeat(train), repeat(test))

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
