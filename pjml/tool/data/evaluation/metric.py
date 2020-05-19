import numpy as np
from sklearn.metrics import accuracy_score

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.lighttransformer import LightTransformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.model.model import Model


class Metric(LightTransformer, FunctionInspector):
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

    def __init__(self, functions=None, target='Y', prediction='Z'):
        if functions is None:
            functions = ['accuracy']
        super().__init__(self._to_config(locals()), deterministic=True)
        self.functions = functions
        self.target, self.prediction = target, prediction
        self.selected = [self.function_from_name[name] for name in functions]

    def _apply_impl(self, data):
        applied = self._use_impl(data, step='a')
        return Model(self, data, applied)

    def _use_impl(self, data, step='u'):
        return data.updated(
            self.transformations(step),
            R=np.array([[f(data, self.target, self.prediction)
                         for f in self.selected]])
        )

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            'function': CatP(choice, items=cls.names()),
            'target': CatP(choice, items=['Y']),
            'prediction': CatP(choice, items=['Z'])
        }
        return TransformerCS(Node(params=params))

    @staticmethod
    def _fun_error(data, target, prediction):
        return 1 - accuracy_score(
            data.field(target, Metric), data.field(prediction, Metric)
        )

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(
            data.field(target, Metric), data.field(prediction, Metric)
        )

    @staticmethod
    def _fun_length(data, target, prediction):
        return len(data.history)
