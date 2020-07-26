from itertools import repeat

from pjdata import types as t
from pjdata.mixin.noinfo import NoInfo
from pjdata.transformer.enhancer import DSStep
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
