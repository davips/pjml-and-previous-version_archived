from functools import lru_cache
from typing import Callable, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score

import pjdata.types as t
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
        print('KWARGS --> ', kwargs)
        if functions is None:
            functions = ["accuracy"]
        super().__init__(self._to_config(locals()), deterministic=True, **kwargs)
        self.functions = functions
        self.target, self.prediction = target, prediction
        self.selected = [self.function_from_name[name] for name in functions]

        print('MeTRICCCCCCCCCCCCCCCCCCCCCCCC')
        print(self._enhance)

    @lru_cache()
    def _enhancer_info(self, train: t.Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        return lambda train: self._transform(train)

    @lru_cache()
    def _model_info(self, test: t.Data) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Data]:
        return lambda test: self._transform(test)

    @lru_cache()
    def _info(self, data: t.Data) -> Dict[str, Any]:
        measures = [[f(data, self.target, self.prediction) for f in self.selected]]
        return {"computed_metric": measures}

    def _transform(self, data) -> t.Data:
        computed_metric = self._info(data)["computed_metric"]
        return data.updated((), R=np.array(computed_metric))

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
