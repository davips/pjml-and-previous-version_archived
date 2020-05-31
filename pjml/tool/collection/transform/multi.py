from functools import lru_cache
from typing import Tuple,Iterator

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.nonfinalizer import NonFinalizer
from pjml.tool.abc.minimalcontainer import TMinimalContainerN
from pjml.tool.abc.mixin.batch import unzip_iterator
from pjml.tool.abc.mixin.component import TComponent


class TMulti(NonFinalizer, TMinimalContainerN):
    """Process each Data object from a collection with its respective
    transformer."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(TMulti.name, TMulti.path, transformers)

    # def iterators(self, train_collection, test_collection) -> Tuple[Iterator]:
    #     funcs = [trf.dual_transform for trf in self.transformers]
    #     iterator = map(
    #         lambda func, prior, posterior: func(prior, posterior),
    #         funcs, train_collection, test_collection
    #     )
    #     return unzip_iterator(iterator)

    @lru_cache()
    def enhancer_info(self):
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    @lru_cache()
    def model_info(self, data):
        return {'models': [trf.model(data) for trf in self.transformers]}

    def enhancer_func(self):
        enhancers = self.enhancer_info['enhancers']

        def transform(collection):
            funcs = [
                lambda data: enhancer.transform(data) for enhancer in
                enhancers
            ]
            iterator = map(
                lambda func, data: func(data), funcs, collection
            )
            return Collection(iterator, lambda: collection.data,
                              debug_info='multi')

        return transform

    def model_func(self, train_collection):
        transformers = self.transformers

        def transform(collection):
            # models = info1['models']
            funcs = [
                lambda prior, posterior: trf.model(prior).transform(
                    posterior)
                for trf in transformers
            ]
            iterator = map(
                lambda func, prior, posterior: func(prior, posterior),
                funcs, train_collection, collection
            )
            return Collection(iterator, lambda: collection.data,
                              debug_info='multi')

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return transform

    @property
    def finite(self):
        return True
