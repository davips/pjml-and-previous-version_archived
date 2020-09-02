from typing import Tuple

from pjdata import types as t
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.containern import ContainerN


class Multi(ContainerN):
    """Process each Data object from a stream with its respective component.

    Container with minimum configuration (seed) for more than one component."""

    # TODO: create stream operator to make __new__ obsolete here?
    def __new__(cls, *args: Component, seed: int = 0, components: Tuple[Component, ...] = None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(t, Component) for t in components]):
            return object.__new__(cls)
        return ContainerCS(Multi.__name__, Multi.__module__, components)

    def __init__(self, *args, seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        outerself = self

        class Enh(Enhancer):
            def _info_impl(self, data):
                return {"enhancers": map(lambda trf: trf.enhancer, outerself.components)}

            def _transform_impl(self, data: t.Data) -> t.Result:
                # noinspection PyUnresolvedReferences
                return {"stream": map(lambda e, d: e.transform(d), self.info(data).enhancers, data.stream)}

        class Mod(Model):
            def _info_impl(self, train):
                return {"models": map(lambda c, d: c.model(d), outerself.components, train.stream)}

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"stream": map(lambda m, d: m.transform(d), self.info.models, data.stream)}

        super().__init__({}, Enh, Mod, seed, components, enhance, model, deterministic=True)
