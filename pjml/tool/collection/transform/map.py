from functools import lru_cache
from typing import Callable, Any, Dict

import pjdata.types as t
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainer1
from pjml.tool.abc.mixin.component import Component


class Map(MinimalContainer1):
    """Execute the same transformer for the entire collection."""

    def __new__(cls, *args, seed=0, transformers=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Component) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Map.name, Map.path, transformers)

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {"enhancer": self.transformer.enhancer}

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {
            "models": map(self.transformer.model, data.stream),   # TODO: decidir sobre lazy models
            # "model_stream": map(self.transformer.model, data.stream)
        }

    def _enhancer_func(self) -> Callable[[t.Data], t.Result]:
        enhancer = self.transformer.enhancer
        return lambda d: {'stream': map(enhancer.transform, d.stream)}

    def _model_func(self, data: t.Data) -> t.Transformation:
        transformer = self.transformer
        return lambda d: {'stream': map(transformer.model(data).transform, d.stream)}
