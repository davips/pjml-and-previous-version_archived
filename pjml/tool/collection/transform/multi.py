import operator
from functools import lru_cache

from itertools import tee
from typing import Optional, List, Tuple, Dict, Iterator

from pjdata.collection import Collection
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer


class Multi(MinimalContainerN):
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

    def iterator(
            self,
            prior_collection: Collection,
            posterior_collection: Collection
    ) -> Iterator:
        funcs = [
            lambda prior, posterior: trf.dual_transform(prior, posterior)
            for trf in self.transformers
        ]
        return map(
            lambda func, prior, posterior: func(prior, posterior),
            funcs, prior_collection, posterior_collection
        )

    def iterators(
            self,
            prior_collection: Collection,
            posterior_collection: Collection
    ) -> Tuple[Iterator, Iterator]:
        gen0, gen1 = tee(
            self.iterator(prior_collection, posterior_collection))

        return map(operator.itemgetter(0), gen0), map(operator.itemgetter(1), gen1)

    def _model_impl(self, prior_collection: Collection) -> Transformer:
        # info1 = self._info1
        transformers = self.transformers

        def transform(collection):
            # models = info1['models']
            funcs = [
                lambda prior, posterior: trf.model(prior).transform(posterior)
                for trf in transformers
            ]
            iterator = map(
                lambda func, prior, posterior: func(prior, posterior),
                funcs, prior_collection, collection
            )
            return Collection(iterator, lambda: collection.data,
                              debug_info='multi')

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return Transformer(
            func=transform,
            info=None  # info1
        )

    @lru_cache()
    def _info2(self) -> Dict[str, List[Transformer]]:
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_impl(self) -> Transformer:
        info2 = self._info2()
        enhancers = info2['enhancers']

        def transform(collection):
            funcs = [
                lambda data: enhancer.transform(data) for enhancer in enhancers
            ]
            iterator = map(
                lambda func, data: func(data), funcs, collection
            )
            return Collection(iterator, lambda: collection.data,
                              debug_info='multi')

            # TODO: Tratar StopException com hint sobre montar better pipeline?

        return Transformer(
            func=transform,
            info=info2,
        )
