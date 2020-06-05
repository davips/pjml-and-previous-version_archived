from functools import lru_cache
from typing import Optional, Tuple, Dict, Callable, Any

from pjdata.aux.util import Property
from pjdata.content.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.nonfinalizer import NonFinalizer


class Multi(NonFinalizer, MinimalContainerN):
    """Process each Data object from a collection with its respective
    transformer."""

    def __new__(
        cls,
        *args: Component,
        seed: int = 0,
        transformers: Optional[Tuple[Component, ...]] = None,
        **kwargs
    ):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Component) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Multi.name, Multi.path, transformers)

    # def iterators(self, train_collection, test_collection) -> Tuple[Iterator]:
    #     funcs = [trf.dual_transform for trf in self.transformers]
    #     iterator = map(
    #         lambda func, prior, posterior: func(prior, posterior),
    #         funcs, train_collection, test_collection
    #     )
    #     return unzip_iterator(iterator)

    @lru_cache()
    def _enhancer_info(self, train_coll=None) -> Dict[str, Any]:
        return {"enhancers": [trf.enhancer for trf in self.transformers]}

    def _enhancer_func(self) -> Callable[[Collection], Collection]:
        enhancers = self._enhancer_info()["enhancers"]

        def transform(train_coll: Collection) -> Collection:
            funcs = [lambda data: enhancer.transform(data) for enhancer in enhancers]
            iterator = map(lambda func, data: func(data), funcs, train_coll)
            return Collection(iterator, lambda: train_coll.data, debug_info="multi")

        return transform

    @lru_cache()
    def _model_info(self, train_coll: Collection) -> Dict[str, Any]:
        models = [
            trf.model(data) for trf, data in zip(self.transformers, train_coll)
        ]  # TODO: aqui trfs acaba antes de coll, então data fica inacessível?
        return {"models": models}

    def _model_func(self, train_coll: Collection) -> Callable[[Collection], Collection]:
        transformers = self.transformers

        def transform(test_coll: Collection) -> Collection:
            # models = info1['models']
            funcs = [
                lambda prior, posterior: trf.model(prior).transform(posterior)
                for trf in transformers
            ]
            iterator = map(
                lambda func, prior, posterior: func(prior, posterior),
                funcs,
                train_coll,
                test_coll,
            )
            return Collection(iterator, lambda: test_coll.data, debug_info="multi")

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return transform

    @Property
    def finite(self) -> bool:
        return True
