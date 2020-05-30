import operator
from functools import lru_cache

from itertools import tee

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import TMinimalContainerN
from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.mixin.transformer import TTransformer


class Multi(TMinimalContainerN):
    """Process each Data object from a collection with its respective
    transformer."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Multi.name, Multi.path, transformers)

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
