import numpy
from numpy import mean

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

    def __init__(self, field='R', function='mean', **kwargs):
        config = self._to_config(locals())
        super().__init__(config, **kwargs, deterministic=True)
        self.function = self.function_from_name[config['function']]
        self.field = field

    def partial_result(self, data):
        return data.field(self.field, 'Summ')

    def final_result_func(self, collection):
        summarize = self.function
        return lambda values: summarize(collection.data, values)

    @classmethod
    def _cs_impl(cls):
        params = {
            'function': CatP(choice, items=cls.function_from_name.keys()),
            'field': CatP(choice, items=['z', 'r', 's'])
        }
        return TransformerCS(nodes=[Node(params)])

    def _fun_mean(self, data, values):
        res = mean([m for m in values], axis=0)
        if isinstance(res, tuple):
            return data.updated(self.transformations('u'), S=numpy.array(res))
        else:
            return data.updated(self.transformations('u'), s=res)
