from itertools import repeat
from typing import Tuple, Iterator, Dict, Any, Callable

from pjdata.content.collection import Collection
from pjdata.content.data import Data
from pjdata.mixin.noinfo import NoInfo
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.nonfinalizer import NonFinalizer


class Repeat(NoInfo, NonFinalizer, Component):
    """Data -> Collection"""

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

    @property
    def finite(self):
        return False

    def _enhancer_func(self) -> Callable[[Data], Collection]:
        return lambda d: Collection(
            repeat(d), lambda: d, finite=False, debug_info="expand"
        )

    def _model_func(self, data: Data) -> Callable[[Data], Collection]:
        return self._enhancer_func()

    def iterator(self, train: Data, test: Data) -> Iterator[Tuple[Data, Data]]:
        return zip(repeat(train), repeat(test))

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
