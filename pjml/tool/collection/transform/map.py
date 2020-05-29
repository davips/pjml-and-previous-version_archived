import operator
from functools import lru_cache
from itertools import tee

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainer1, \
    TMinimalContainer1
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
            model = self.transformer.apply(data,
                                           exit_on_error=self._exit_on_error)
            datas.append(model.data)
            models.append(model)
        applied = collection.updated(self.transformations(step='a'),
                                     datas=datas)
        # TODO: which containers should pass self._exit_on_error to transformer?
        return ContainerModel(self, collection, applied, models)

    def _use_impl(self, collection, models=None):
        size = len(models)
        if size != collection.size:
            raise Exception('Collections passed to apply and use should have '
                            f'the same size a- {size} != u- {collection.size}')
        datas = []
        for model in models:
            data = model.use(next(collection),
                             exit_on_error=self._exit_on_error)
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

    def iterator(self, prior_collection, posterior_collection):
        return map(
            self.transformer.dual_transform,
            prior_collection, posterior_collection
        )

    def iterators(self, prior_collection, posterior_collection):
        gen0, gen1 = tee(
            self.iterator(prior_collection, posterior_collection))
        return map(operator.itemgetter(0), gen0), \
               map(operator.itemgetter(1), gen1)

    # @lru_cache()
    # def _info(self, prior_collection):
    #     return {
    #         'models': [self.transformer.model(data)
    #                    for data in prior_collection]
    #     }

    def _model_impl(self, prior_collection):
        # info1 = self._info(prior_collection)
        # models = info1['models']
        transformer = self.transformer

        def transform(posterior_collection):
            def func(prior, posterior):
                return transformer.model(prior).transform(posterior)

            iterator = map(func, prior_collection, posterior_collection)
            return Collection(iterator, lambda: posterior_collection.data,
                              debug_info='map')

        return TTransformer(
            func=transform,
            info=None  # info1
        )

    @lru_cache()
    def _info2(self):
        return {'enhancer': self.transformer.enhancer}

    def _enhancer_impl(self):
        info2 = self._info2()
        enhancer = info2['enhancer']

        def transform(posterior_collection):
            iterator = map(enhancer.transform, posterior_collection)
            return Collection(iterator, lambda: posterior_collection.data,
                              debug_info='map')

        return TTransformer(
            func=transform,
            info=info2
        )
