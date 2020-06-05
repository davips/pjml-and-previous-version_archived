from typing import Callable, Any, Dict

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainer1
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.nonfinalizer import NonFinalizer


class Map(NonFinalizer, MinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Component) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Map.name, Map.path, transformers)

    def _enhancer_info(self, train_coll: Collection) -> Dict[str, Any]:
        return {"enhancer": self.transformer.enhancer}

    def _model_info(self, train_coll: Collection) -> Dict[str, Any]:
        return {"models": [self.transformer.model(data) for data in train_coll]}

    def _enhancer_func(self) -> Callable[[Collection], Collection]:
        enhancer = self.transformer.enhancer

        def transform(test_coll: Collection) -> Collection:
            iterator = map(enhancer.transform, test_coll)
            return Collection(iterator, lambda: test_coll.data, debug_info="map")

        return transform

    def _model_func(self, train_coll: Collection) -> Callable[[Collection], Collection]:
        transformer = self.transformer

        def transform(test_coll: Collection) -> Collection:
            def func(train, test):
                return transformer.model(train).transform(test)

            iterator = map(func, train_coll, test_coll)
            return Collection(iterator, lambda: test_coll.data, debug_info="map")

        return transform

    @property
    def finite(self) -> bool:
        return True

    # def iterators(self, train_collection, test_collection):
    #     iterator = map(self.transformer.dual_transform,
    #                    train_collection, test_collection)
    #     return unzip_iterator(iterator)
