from functools import lru_cache
from typing import Optional, Tuple, Dict, Callable, Any

from pjdata import types as t
from pjdata.aux.util import Property
from pjdata.content.data import Data
from pjdata.types import Result
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component
from pjml.tool.collection.accumulator import Accumulator


class Multi(MinimalContainerN):
    """Process each Data object from a stream with its respective transformer."""

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

    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {"enhancers": map(lambda trf: trf.enhancer, self.transformers)}

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        models = map(lambda trf, d: trf.model(d), self.transformers, data.stream)
        return {"models": models}

    def _enhancer_func(self) -> t.Transformation:
        enhancers = self._enhancer_info()["enhancers"]
        return lambda data: {
            'stream': map(lambda e, d: e.transform(d), enhancers, data.stream)
        }

    def _model_func(self, data: Data) -> Callable[[Data], Result]:
        models = self._model_info(data)["models"]
        return lambda test: {
            'stream': map(lambda m, d: m.transform(d), models, test.stream)
        }
