from typing import Iterable

import numpy
from numpy import ndarray, mean

from pjdata import types as t
from pjdata.transformer.enhancer import DSStep
from pjml.config.description.cs.cs import CS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.functioninspection import withFunctionInspection
from pjml.tool.abs.mixin.noinfo import withNoInfo
from pjml.tool.stream.reduce.accumulator import Accumulator


class InterruptedStreamException(Exception):
    pass


class Summ(Component, withFunctionInspection):
    """Given a field, summarizes a Collection object to a Data object.

    The resulting Data object will have only the 's' field. To keep other
    fields, consider using a Keep containing all the concurrent part:
    Keep(Expand -> ... -> Summ).

    The cells of the given field (matrix) will be averaged across all data
    objects, resulting in a new matrix with the same dimensions.
    """

    def __init__(self, field: str = "R", function: str = "mean", **kwargs):
        config = self._to_config(locals())
        self.function = Summ.function_from_name()[config["function"]]
        self.field = field
        summarize = self.function
        outerself = self

        class Step(withNoInfo, DSStep):
            def _transform_impl(self, data: t.Data) -> t.Result:
                def step(d, acc):
                    if d.isfrozen or d.failure:
                        return d.transformedby(self.pholder), None
                    acc.append(d.field(outerself.field, "Summ"))
                    return d, acc

                iterator = Accumulator(data.stream, start=[], step_func=step, summ_func=summarize)

                def lazy():
                    # try:
                    return iterator.result

                # except StreamException:

                return {"stream": iterator, "S": lazy}

        super().__init__(config, enhancer_cls=Step, model_cls=Step, deterministic=True, **kwargs)

    @classmethod
    def _cs_impl(cls) -> CS:
        params = {
            "function": CatP(choice, items=cls.function_from_name().keys()),
            "field": CatP(choice, items=["z", "r", "s"]),
        }
        return CS(nodes=[Node(params)])

    @staticmethod
    def _fun_mean(values: Iterable[float]) -> ndarray:
        res = mean([m for m in values], axis=0)
        return numpy.array(res) if isinstance(res, tuple) else res
