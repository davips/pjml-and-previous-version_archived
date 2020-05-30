import operator
from itertools import tee

import numpy
from numpy import mean

from pjdata.collection import Collection, AccResult
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.mixin.transformer import Transformer


class RSumm(Component, FunctionInspector):
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

    def make_iterator(self, prior_collection, posterior_collection):
        def func(prior, posterior):
            if self.onenhancer:
                prior = self.enhancer.transform(prior)
            if self.onmodel:
                posterior = self.model(prior).transform(posterior)
            return prior, posterior

        return map(func, prior_collection, posterior_collection)

    def iterators(self, prior_collection, posterior_collection):
        gen0, gen1 = tee(
            self.make_iterator(prior_collection, posterior_collection))
        return map(operator.itemgetter(0), gen0), \
               map(operator.itemgetter(1), gen1)

    def _enhancer_impl(self):
        field_name = self.field

        def transform(collection):
            def finalize(values):
                print('finalizing summmmmmmmmmmmmmmmm')
                return self.function(collection.data, values)

            def iterator():
                acc = []
                print('\nSumm start iterator...')
                for data in collection:
                    acc.append(data.field(field_name, 'Summ'))
                    yield AccResult(data, acc)
                print('...Summ finish iterator.\n')
            return Collection(iterator(), finalize, debug_info='summ')

        return Transformer(
            func=transform,
            info=None
        )

    def _model_impl(self, prior):
        return self._enhancer_impl()

    def transformations(self, step, clean=True):
        return super().transformations('u')

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
