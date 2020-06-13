from abc import abstractmethod
from collections import Iterator
from functools import lru_cache
from typing import Callable, Iterable, Dict, Any, List, Generator

import numpy
from numpy import ndarray, mean

from pjdata.content.data import Data
from pjdata.types import Result, Field
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.collection.accumulator import Accumulator


class Summ(Component, FunctionInspector):
    """Given a field, summarizes a Collection object to a Data object.

    The resulting Data object will have only the 's' field. To keep other
    fields, consider using a Keep containing all the concurrent part:
    Keep(Expand -> ... -> Summ).

    The collection history will be exported to the summarized Data object.

    The cells of the given field (matrix) will be averaged across all data
    objects, resulting in a new matrix with the same dimensions.
    """

    def __init__(self, field: str = "R", function: str = "mean", **kwargs):
        config = self._to_config(locals())
        super().__init__(config, deterministic=True, **kwargs)
        self.function = self.function_from_name[config["function"]]
        self.field = field

    @lru_cache()
    def _enhancer_info(self, data: Data = None) -> Dict[str, Any]:
        return {}

    @lru_cache()
    def _model_info(self, data: Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[Data], Result]:
        summarize = self.function

        def transform(data: Data) -> Result:
            def step(d, acc):
                acc.append(d.field(self.field, "Summ"))
                return d, acc

            iterator = Accumulator(data.stream, start=[], step_func=step, summ_func=summarize)

            return {'stream': iterator, 'S': lambda: iterator.result}

        return transform

    def _model_func(self, data: Data) -> Callable[[Data], Result]:
        return self._enhancer_func()

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        params = {
            "function": CatP(choice, items=cls.function_from_name.keys()),
            "field": CatP(choice, items=["z", "r", "s"]),
        }
        return TransformerCS(nodes=[Node(params)])

    @staticmethod
    def _fun_mean(values: Iterable[float]) -> ndarray:
        res = mean([m for m in values], axis=0)
        return numpy.array(res) if isinstance(res, tuple) else res
