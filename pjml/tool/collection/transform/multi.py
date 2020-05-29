import operator
from functools import lru_cache
from itertools import cycle, tee

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN, \
    TMinimalContainerN
from pjml.tool.abc.mixin.component import TTransformer, TComponent
from pjml.tool.abc.transformer import UTransformer
from pjml.tool.model.containermodel import ContainerModel


# class Multi(MinimalContainerN):
#     """Process each Data object from a collection with its respective
#     transformer."""
#
#     def __new__(cls, *args, transformers=None, seed=0):
#         """Shortcut to create a ConfigSpace."""
#         if transformers is None:
#             transformers = args
#         if all([isinstance(t, UTransformer) for t in transformers]):
#             return object.__new__(cls)
#         return ContainerCS(Multi.name, Multi.path, transformers)
#
#     def _apply_impl(self, collection):
#         # TODO: somehow enforce collection for all concurrent components.
#         if collection.isfinite and self.size != collection.size:
#             raise Exception(
#                 f'Config space and collection should have the same size '
#                 f'{self.size} != collection {collection.size}'
#             )
#         models = []
#         datas = []
#         for transformer in self.transformers:
#             model = transformer.apply(
#                 next(collection),
#                 exit_on_error=self._exit_on_error
#             )
#             datas.append(model.data)
#             models.append(model)
#
#         applied = collection.updated(
#             self.transformations('a'), datas=datas
#         )
#         return ContainerModel(self, collection, applied, models)
#
#     def _use_impl(self, collection, models=None):
#         if collection.isfinite and self.size != collection.size:
#             raise Exception(
#                 'Config space and collection should have the same '
#                 f'size {self.size} != collection {collection.size}'
#             )
#         datas = []
#         for model in models:
#             data = model.use(
#                 next(collection),
#                 exit_on_error=self._exit_on_error
#             )
#             datas.append(data)
#         return collection.updated(self.transformations('u'), datas=datas)


class TMulti(TMinimalContainerN):
    """Process each Data object from a collection with its respective
    transformer."""

    def __new__(cls, *args, transformers=None, seed=0):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(TMulti.name, TMulti.path, transformers)

    @lru_cache()
    def _info1(self, prior_collection):
        models = []
        raise NotImplementedError
        # return {'models': models}

    def iterator(self, prior_collection, posterior_collection):
        # print('mul',self.transformers)
        funcs = [
            lambda prior, posterior: trf.dual_transform(prior, posterior)
            for trf in self.transformers
        ]
        return map(
            lambda func, prior, posterior: func(prior, posterior),
            funcs, prior_collection, posterior_collection
        )

    def iterators(self, prior_collection, posterior_collection):
        gen0, gen1 = tee(
            self.iterator(prior_collection, posterior_collection))
        return map(operator.itemgetter(0), gen0), \
               map(operator.itemgetter(1), gen1)

    def _model_impl(self, prior_collection):
        # info1 = self._info1
        transformers = self.transformers

        def transform(posterior_collection):
            # models = info1['models']
            funcs = [
                lambda prior, posterior: trf.model(prior).transform(posterior)
                for trf in transformers
            ]
            iterator = map(
                lambda func, prior, posterior: func(prior, posterior),
                funcs, prior_collection, posterior_collection
            )
            return Collection(iterator, lambda: posterior_collection.data,
                              debug_info='multi')

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return TTransformer(
            func=transform,
            info=None  # info1
        )

    @lru_cache()
    def _info2(self):
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_impl(self):
        info2 = self._info2()
        enhancers = info2['enhancers']

        def transform(posterior_collection):
            funcs = [
                lambda data: enhancer.transform(data) for enhancer in enhancers
            ]
            iterator = map(
                lambda func, data: func(data), funcs, posterior_collection
            )
            return Collection(iterator, lambda: posterior_collection.data,
                              debug_info='multi')

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return TTransformer(
            func=transform,
            info=info2,
        )
