from functools import lru_cache
from typing import Callable, Any, Dict

import pjdata.types as t
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abs.minimalcontainer import MinimalContainer1
from pjml.tool.abs.component import Component


class Map(MinimalContainer1):
    """Execute the same component for the entire stream."""

    def __new__(cls, *args, seed=0, components=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(c, Component) for c in components]):
            return object.__new__(cls)
        return ContainerCS(Map.__name__, Map.__module__, components)

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:  #TODO: should _*info accept None?
        return {"enhancer": self.component.enhancer}

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {"models": map(self.component.model, data.stream)}

    def _enhancer_func(self) -> Callable[[t.Data], t.Result]:
        enhancer = self.component.enhancer
        return lambda d: {'stream': map(enhancer.transform, d.stream)}

    def _model_func(self, data: t.Data) -> t.Transformation:
        component = self.component
        return lambda d: {'stream': map(component.model(data).transform, d.stream)}
