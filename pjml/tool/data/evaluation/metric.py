from functools import lru_cache
from typing import Callable, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score

import pjdata.types as t
from pjdata.transformer.enhancer import Enhancer, DSStep
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.cs import CS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.functioninspection import withFunctionInspection
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Metric(Component, withFunctionInspection):
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
        cfg = self._to_config(locals())
        outerself = self

        class Step(withNoInfo, DSStep):
            def _transform_impl(self, data):
                return {"R": np.array([[f(data, outerself.target, outerself.prediction) for f in outerself.selected]])}

        super().__init__(cfg, Step, Step, deterministic=True, **kwargs)
        self.functions = functions
        self.target, self.prediction = target, prediction
        self.selected = [Metric.function_from_name()[name] for name in functions]

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            "function": CatP(choice, items=cls.names()),
            "target": CatP(choice, items=["Y"]),
            "prediction": CatP(choice, items=["Z"]),
        }
        return CS(nodes=[Node(params=params)])

    @staticmethod
    def _fun_error(data, target, prediction):
        return 1 - accuracy_score(data.field(target, "metric"), data.field(prediction, "metric"))

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(data.field(target, "metric"), data.field(prediction, "metric"))

    @staticmethod
    def _fun_length(data, target, prediction):
        return len(data.history)
