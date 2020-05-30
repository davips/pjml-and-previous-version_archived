import operator
from functools import lru_cache

from itertools import tee

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import TMinimalContainer1
from pjml.tool.abc.mixin.component import TTransformer, TComponent


class TMap(TMinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(TMap.name, TMap.path, transformers)

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
