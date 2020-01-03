from functools import lru_cache

from sklearn.metrics import accuracy_score

from pjml.config.cs.configspace import ConfigSpace
from pjml.config.distributions import choice
from pjml.config.parameter import CatP
from pjml.tool.base.aux.functioninspector import FunctionInspector
from pjml.tool.base.transformer import Transformer


class Metric(Transformer, FunctionInspector):
    """Metric to evaluate a given Data field.

    Developer: new metrics can be added just following the pattern '_fun_xxxxx'
    where xxxxx is the name of the new metric.

    Parameters
    ----------
    function
        Name of the function to use to evaluate data objects.
    target
        Name of the matrix with expected values.
    prediction
        Name of the matrix to be evaluated.
    """

    def __init__(self, function, target='Y', prediction='Z'):
        super().__init__(self._to_config(locals()),
                         self.functions[function],
                         deterministic=True)
        self.target, self.prediction = target, prediction
        self.model = self.algorithm

    def _apply_impl(self, data):
        return self._use_impl(data)

    def _use_impl(self, data):
        return data.updated1(self._transformation(), r=self.algorithm(data))

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            'function': CatP(choice, items=cls.functions.keys())
        }
        return ConfigSpace(params=params)

    def _fun_error(self, data):
        return 1 - accuracy_score(
            data.matrices[self.target], data.matrices[self.prediction]
        )

    def _fun_accuracy(self, data):
        return accuracy_score(
            data.matrices[self.target], data.matrices[self.prediction]
        )


