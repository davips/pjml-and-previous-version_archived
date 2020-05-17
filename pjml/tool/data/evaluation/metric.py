import numpy as np
from sklearn.metrics import accuracy_score

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.component import TComponent, TTransformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import ISTransformer


class Metric(ISTransformer, FunctionInspector):
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

    # def _apply_impl(self, data):
    #     applied = self._use_impl(data, step='a')
    #     return Model(self, data, applied)

    def _use_impl(self, data, step='u'):
        if self.target not in data.matrices:
            print(f'Impossible to calculate metric {self.functions}: \n'
                  f'Field {self.target} does not exist!\nAvailable fields:',
                  data.field_names)
            raise Exception('Missing field!')

        if self.prediction not in data.matrices:
            raise Exception(
                f'Impossible to calculate metric {self.functions}: Field '
                f'{self.prediction} does not exist!')
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
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_length(data, target, prediction):
        return len(data.history)


class TMetric(TComponent, FunctionInspector):
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

    def __init__(self, functions=None, target='Y', prediction='Z', **kwargs):
        if functions is None:
            functions = ['accuracy']
        super().__init__(self._to_config(locals()), deterministic=True, **kwargs)
        self.functions = functions
        self.target, self.prediction = target, prediction
        self.selected = [self.function_from_name[name] for name in functions]

    def _modeler_impl(self, prior):
        return self._enhancer_impl()

    def _enhancer_impl(self):
        def func(posterior):
            return self._func(posterior)
        return TTransformer(func=func)

    def _func(self, data, step='u'):
        if self.target not in data.matrices:
            print(f'Impossible to calculate metric {self.functions}: \n'
                  f'Field {self.target} does not exist!\nAvailable fields:',
                  data.field_names)
            raise Exception('Missing field!')

        if self.prediction not in data.matrices:
            raise Exception(
                f'Impossible to calculate metric {self.functions}: Field '
                f'{self.prediction} does not exist!')
        print(123123123123123123123)
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
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_accuracy(data, target, prediction):
        return accuracy_score(
            data.matrices[target], data.matrices[prediction]
        )

    @staticmethod
    def _fun_length(data, target, prediction):
        return len(data.history)

