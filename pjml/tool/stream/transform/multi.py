from typing import Tuple, Dict, Callable, Any

from pjdata import types as t
from pjdata.content.data import Data
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.transformer import Transformer
from pjdata.types import Result
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.minimalcontainer import MinimalContainerN


class Multi(MinimalContainerN):
    """Process each Data object from a stream with its respective component."""

    def __new__(cls, *args: Component, seed: int = 0, components: Tuple[Component, ...] = None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(t, Component) for t in components]):
            return object.__new__(cls)
        return ContainerCS(Multi.__name__, Multi.__module__, components)

    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {"enhancers": map(lambda trf: trf.enhancer, self.components)}

    def _enhancer_impl(self) -> Transformer:
        enhancers = self._enhancer_info()["enhancers"]
        return Enhancer(self,
                        lambda data: {"stream": map(lambda e, d: e.transform(d), enhancers, data.stream)},
                        lambda _: {"enhancers": enhancers}
                        )

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        models = map(lambda c, d: c.model(d), self.components, data.stream)
        return {"models": models}

    def _model_impl(self, data: t.Data) -> Transformer:
        models = self._model_info(data)["models"]
        return Model(self,
                     lambda test: {"stream": map(lambda m, d: m.transform(d), models, test.stream)}, {"models": models},
                     data
                     )
