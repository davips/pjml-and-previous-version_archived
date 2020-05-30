import operator
from functools import reduce
from itertools import accumulate, repeat, tee

import numpy
from numpy import mean
from numpy import std

from pjdata.collection import Collection, AccResult
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.abc.mixin.component import TTransformer, TComponent
from pjml.tool.abc.mixin.functioninspector import FunctionInspector


class TRSumm(TComponent, FunctionInspector):
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
        super().__init__(config, deterministic=True, **kwargs)
        self.function = self.function_from_name[config['function']]
        self.field = field

    def iterator(self, prior_collection, posterior_collection):
        def func(prior, posterior):
            return self.model(prior).transform(posterior), \
                   self.enhancer.transform(posterior)

        return map(func, prior_collection, posterior_collection)

    def iterators(self, prior_collection, posterior_collection):
        gen0, gen1 = tee(
            self.iterator(prior_collection, posterior_collection))
        return map(operator.itemgetter(0), gen0), \
               map(operator.itemgetter(1), gen1)

    def dual_transform(self, prior_collection, posterior_collection):
        print(self.__class__.__name__, ' dual transf (((')
        field_name = self.field

        def finalize(data, values):
            return self.function(data, values)

        def iterator(collection):
            acc = []
            for data in collection:
                acc.append(data.field(field_name, 'Summ'))
                yield AccResult(data, acc)
            print('...Summ finish iterator.\n')

        return (
            Collection(
                iterator(prior_collection),
                lambda values: finalize(prior_collection.data, values),
                debug_info= self.__class__.__name__ + ' pri'
            ),
            Collection(
                iterator(posterior_collection),
                lambda values: finalize(posterior_collection.data, values),
                debug_info= self.__class__.__name__ + ' pos'
            )
        )

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

            # Versão obscura:
            # print('\nSumm start iterator...')
            # def func(t1, t2):
            #     _, acc0 = t1
            #     data, _ = t2
            #     acc0.append(data.field(field_name, 'Summ'))
            #     return data, acc0
            #
            # iterator = accumulate(zip(collection, repeat(None)),
            #                        func, initial=(None, []))
            # # Discards initial value.
            # next(iterator)
            # print('...Summ finish iterator.\n')

            return Collection(iterator(), finalize, debug_info='summ')

        return TTransformer(
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

# def _fun_std(self, collection):
#     return std([data.field(self.field, self) for data in collection],
#                axis=0)
#
# def _fun_mean_std(self, collection):
#     # TODO?: optimize calculating mean and stdev together
#     values = [data.field(self.field, self) for data in collection]
#     if len(values[0].shape) == 2:
#         if values[0].shape[0] > 1:
#             raise Exception(
#                 f"Summ doesn't accept multirow fields: {self.field}\n"
#                 f"Shape: {values[0].shape}")
#         values = [v[0] for v in values]
#     return mean(values, axis=0), std(values, axis=0)
