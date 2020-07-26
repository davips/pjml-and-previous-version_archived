from functools import lru_cache
from typing import Callable, Any, Dict

import pjdata.types as t
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.transformer import Transformer
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
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {"models": map(self.component.model, data.stream)}

    def _enhancer_impl(self) -> Transformer:
        enhancer = self.component.enhancer
        return Enhancer(self,
                        lambda d: {"stream": map(enhancer.transform, d.stream)}, lambda _: {"enhancer": enhancer}
                        )

    def _model_impl(self, data: t.Data) -> Transformer:
        models = self._model_info(data)["models"]
        component = self.component
        return Model(self,
                     lambda d: {"stream": map(component.model(data).transform, d.stream)}, {"models": models}, data
                     )
