from itertools import repeat
from typing import Callable

from pjdata import types as t
from pjdata.content.data import Data
from pjdata.mixin.noinfo import NoInfo
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.transformer import Transformer
from pjdata.types import Result
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component


class Repeat(NoInfo, Component):
    """Add infinite stream to a Data object."""

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

    def _enhancer_impl(self) -> Transformer:
        return Enhancer(self, lambda d: {"stream": repeat(d)}, lambda _: {})

    def _model_impl(self, data: t.Data) -> Transformer:
        return Enhancer(self, lambda d: {"stream": repeat(d)}, lambda _: {})

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
