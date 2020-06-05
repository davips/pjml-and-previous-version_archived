from typing import Callable, Iterable, Dict, Any

import numpy
from numpy import mean

from pjdata.content.collection import Collection
from pjdata.content.data import Data
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.finalizer import Finalizer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector


class RSumm(Finalizer, FunctionInspector):
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

    def _enhancer_info(self, data: Collection) -> Dict[str, Any]:
        return {}

    def _model_info(self, data: Collection) -> Dict[str, Any]:
        return {}

    def partial_result(self, data: Data) -> Data:
        return data.field(self.field, "Summ")

    def final_result_func(self, collection: Collection) -> Callable[[Iterable], Data]:
        summarize = self.function
        return lambda values: summarize(collection.data, values)

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        params = {
            "function": CatP(choice, items=cls.function_from_name.keys()),
            "field": CatP(choice, items=["z", "r", "s"]),
        }
        return TransformerCS(nodes=[Node(params)])

    def _fun_mean(self, data: Data, values: Iterable) -> Data:
        res = mean([m for m in values], axis=0)
        if isinstance(res, tuple):
            return data.updated(self.transformations("u"), S=numpy.array(res))
        else:
            return data.updated(self.transformations("u"), s=res)
