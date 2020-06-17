from itertools import repeat
from typing import Callable

from pjdata.content.data import Data
from pjdata.mixin.noinfo import NoInfo
from pjdata.types import Result
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.mixin.component import Component


class Repeat(NoInfo, Component):
    """Add infinite stream to a Data object."""

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

    def _enhancer_func(self) -> Callable[[Data], Result]:
        return lambda d: {'stream': repeat(d)}

    def _model_func(self, data: Data) -> Callable[[Data], Result]:
        return self._enhancer_func()

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
