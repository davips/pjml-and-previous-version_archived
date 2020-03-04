import numpy as np
from sklearn.metrics import accuracy_score

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import Transformer


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

    def __init__(self, function=['accuracy'], target='Y', prediction='Z'):
        super().__init__(self._to_config(locals()), function,
                         deterministic=True)
        self.target, self.prediction = target, prediction
        self.collection_function = self.model = [self.functions[alg_str]
                                                 for alg_str in self.algorithm]
        self.function_name = function

    def _apply_impl(self, data):
        return self._use_impl(data)

    def _use_impl(self, data):
        if self.target not in data.matrices:
            raise Exception(
                f'Impossible to calculate metric {self.function_name}: Field '
                f'{self.target} does not exist!')
        if self.prediction not in data.matrices:
            raise Exception(
                f'Impossible to calculate metric {self.function_name}: Field '
                f'{self.prediction} does not exist!')
        return data.updated(
            self.transformations(),
            R=np.array([[function(data, self.target, self.prediction)
               for function in self.collection_function]])
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
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_length(data, target, prediction):
        return data.history.size