from functools import lru_cache

from pjdata.infinitecollection import InfiniteCollection

from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainer1, TMinimalContainer1
from pjml.tool.abc.mixin.component import TTransformer, TComponent
from pjml.tool.abc.transformer import UTransformer
from pjml.tool.model.containermodel import ContainerModel


class Map(MinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, transformers=None, seed=0):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, UTransformer) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Map.name, Map.path, transformers)

    def _apply_impl(self, collection):
        if not collection.isfinite:
            raise Exception('Collection should be finite for Map!')
        models = []
        datas = []
        for data in collection:
            model = self.transformer.apply(data, exit_on_error=self._exit_on_error)
            datas.append(model.data)
            models.append(model)
        applied = collection.updated(self.transformations(step='a'), datas=datas)
        # TODO: which containers should pass self._exit_on_error to transformer?
        return ContainerModel(self, collection, applied, models)

    def _use_impl(self, collection, models=None):
        size = len(models)
        if size != collection.size:
            raise Exception('Collections passed to apply and use should have '
                            f'the same size a- {size} != u- {collection.size}')
        datas = []
        for model in models:
            data = model.use(next(collection), exit_on_error=self._exit_on_error)
            datas.append(data)
        return collection.updated(self.transformations(step='u'), datas=datas)


class TMap(TMinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, transformers=None, seed=0):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Map.name, Map.path, transformers)

    @lru_cache()
    def _info(self, prior_collection):
        return [self.transformer.modeler(data) for data in prior_collection]

    def _modeler_impl(self, prior_collection):
        models = self._info(prior_collection)
        datas = []

        def func(posterior_collection):
            for model in models:
                datas.append(model.transform(next(posterior_collection)))
            return posterior_collection.updated(
                self.transformations(step='a'),
                datas=datas
            )
        return TTransformer(func=func, models=models)

    @lru_cache()
    def _info2(self):
        return [trf.enhancer() for trf in self.transformers]

    def _enhancer_impl(self):
        def func(posterior_collection):
            # enhancers = self._info2()
            # size = len(enhancers)
            # print(1111111111111111111111, size)
            # print(2222222222222222222222, posterior_collection.size)
            # if size != posterior_collection.size:
            #     raise Exception(
            #         'Collections passed to apply and use should have '
            #         f'the same size a- {size} != u- {posterior_collection.size}'
            #     )
            datas = []
            for data in posterior_collection:
                datas.append(self.transformer.enhancer().transform(data))
            return posterior_collection.updated(
                self.transformations(step='u'), datas=datas
            )
        return TTransformer(func=func)
