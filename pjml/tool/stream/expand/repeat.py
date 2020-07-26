from itertools import repeat
from typing import Callable

from pjdata import types as t
from pjdata.content.data import Data
from pjdata.mixin.noinfo import NoInfo
from pjdata.transformer.enhancer import Enhancer, DSStep
from pjdata.transformer.transformer import Transformer
from pjdata.types import Result
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Repeat(NoInfo, Component):
    """Add infinite stream to a Data object."""

    def __init__(self, **kwargs):
        class Step(withNoInfo, DSStep):

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"stream": repeat(data)}

        super().__init__({}, enhancer_cls=Step, model_cls=Step, deterministic=True, **kwargs)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()
