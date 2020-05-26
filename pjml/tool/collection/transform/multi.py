from functools import lru_cache

from pjdata.finitecollection import FiniteCollection

from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN, TMinimalContainerN
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

        # TODO: Tratar StopException com hint sobre montar better pipeline?
        for trf in self.transformers:
            data = next(prior_collection)
            if data is None:
                raise Exception('Less Data objects than expected!')
            yield trf.model(prior_collection).transform(data)

        # Sentinel marking end of transformed stream.
        yield None

        # Unchanged main data.
        yield next(prior_collection)

        return {'models': models}

    def _model_impl(self, prior_collection):
        info1 = self._info1
        transformers = self.transformers

        def transform(posterior_collection):
            # enhancers = info1['models']

            # TODO: Tratar StopException com hint sobre montar better pipeline?
            for trf in transformers:
                data = next(posterior_collection)
                if data is None:
                    raise Exception('Less Data objects than expected!')
                yield trf.model(next(prior_collection)).transform(data)

            # Sentinel marking end of transformed stream.
            yield None

            # Unchanged main data.
            yield next(posterior_collection)


        return TTransformer(
            func=transform,
            info=info1
        )

    @lru_cache()
    def _info2(self):
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_impl(self):
        info2 = self._info2()

        def transform(prior_collection):
            enhancers = info2['enhancers']

            # TODO: Tratar StopException com hint sobre montar better pipeline?
            for enhancer in enhancers:
                data = next(prior_collection)
                if data is None:
                    raise Exception('Less Data objects than expected!')
                yield enhancer.transform(data)

            # Sentinel marking end of transformed stream.
            yield None

            # Unchanged main data.
            yield next(prior_collection)

        return TTransformer(
            func=transform,
            info=info2,
        )
