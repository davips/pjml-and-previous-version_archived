import numpy as np

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import ISTransformer


class Calc(ISTransformer, FunctionInspector):
    """Calc to evaluate a given Data field.

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

    def __init__(self, functions=None, input_field='S', output_field='S'):
        if functions is None:
            functions = ['mean']
        super().__init__(self._to_config(locals()), deterministic=True)
        self.input_field, self.output_field = input_field, output_field
        self.selected = [self.function_from_name[name] for name in functions]
        self.functions = functions

    # def _apply_impl(self, data):
    #     applied = self._use_impl(data, step='a')
    #     return Model(self, data, applied)

    def _use_impl(self, data, step='u'):
        if self.input_field not in data.matrices:
            raise Exception(
                f'Impossible to calculate {self.functions}: Field '
                f'{self.input_field} does not exist!')

        result_vectors = [function(data.field(self.input_field, self))
                          for function in self.selected]
        dic = {self.output_field: np.array(result_vectors)}
        return data.updated(self.transformations(step), **dic)

    @classmethod
    def _cs_impl(cls):
        # TODO target and prediction
        params = {
            'function': CatP(choice, items=cls.names()),
            'input_field': CatP(choice, items=['S']),
            'output_field': CatP(choice, items=['S'])
        }
        return TransformerCS(Node(params=params))

    @staticmethod
    def _fun_mean(input):
        return np.mean(input)

    @staticmethod
    def _fun_flatten(input):
        return input.flatten()
