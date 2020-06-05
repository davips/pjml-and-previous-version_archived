from functools import lru_cache
from typing import Callable, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score

from pjdata.aux.util import DataT
from pjdata.data import Data
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.functioninspector import FunctionInspector


class Metric(Component, FunctionInspector):
    """Metric to evaluate a given Data field.

    Developer: new metrics can be added just following the pattern '_fun_xxxxx'
    where xxxxx is the name of the new metric.

    Parameters
    ----------
    functions
        Name of the function to use to evaluate data objects.
    target
        Name of the matrix with expected values.
    prediction
        Name of the matrix to be evaluated.
    """

    def __init__(self, functions=None, target="Y", prediction="Z", **kwargs):
        if functions is None:
            functions = ["accuracy"]
        super().__init__(self._to_config(locals()), deterministic=True, **kwargs)
        self.functions = functions
        self.target, self.prediction = target, prediction
        self.selected = [self.function_from_name[name] for name in functions]

    def _enhancer_info(self, train: DataT) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[DataT], DataT]:
        return lambda train: self._transform(train)

    def _model_info(self, test: DataT) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: Data) -> Callable[[Data], Data]:
        return lambda test: self._transform(test)

    @lru_cache()
    def _info(self, data: Data):
        measures = [[f(data, self.target, self.prediction) for f in self.selected]]
        return {"computed_metric": measures}

    def _transform(self, data, step="u"):
        computed_metric = self._info(data)["computed_metric"]
        return data.updated(self.transformations(step), R=np.array(computed_metric))

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            "function": CatP(choice, items=cls.names()),
            "target": CatP(choice, items=["Y"]),
            "prediction": CatP(choice, items=["Z"]),
        }
        return TransformerCS(Node(params=params))

    @staticmethod
    def _fun_error(data, target, prediction):
        return 1 - accuracy_score(
            data.field(target, "metric"), data.field(prediction, "metric")
        )

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(
            data.field(target, "metric"), data.field(prediction, "metric")
        )

    @staticmethod
    def _fun_length(data, target, prediction):
        return len(data.history)
