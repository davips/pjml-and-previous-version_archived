import operator
from functools import lru_cache

from itertools import tee

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import TMinimalContainer1
from pjml.tool.abc.mixin.batch import unzip_iterator
from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.nonfinalizer import NonFinalizer
from pjml.tool.abc.minimalcontainer import MinimalContainer1
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer


class MMap(NonFinalizer, TMinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Component) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Map.name, Map.path, transformers)

    def enhancer_info(self):
        return {'enhancer': self.transformer.enhancer}

    def model_info(self, collection):
        return {
            'models': [self.transformer.model(data) for data in collection]
        }

    def enhancer_func(self):
        enhancer = self.transformer.enhancer

        def transform(collection):
            iterator = map(enhancer.transform, collection)
            return Collection(iterator, lambda: collection.data,
                              debug_info='map')

        return transform

    def model_func(self, train_collection):
        transformer = self.transformer

        def transform(collection):
            def func(train, test):
                return transformer.model(train).transform(test)

            iterator = map(func, train_collection, collection)
            return Collection(iterator, lambda: collection.data,
                              debug_info='map')

        return transform

    @property
    def finite(self):
        return True

    # def iterators(self, train_collection, test_collection):
    #     iterator = map(self.transformer.dual_transform,
    #                    train_collection, test_collection)
    #     return unzip_iterator(iterator)
