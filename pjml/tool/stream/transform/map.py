import pjdata.types as t
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.container1 import Container1


class Map(Container1):
    """Execute the same component for the entire stream.

    Container with minimum configuration (seed) for a single component.
    If more are given, they will be handled as a single Chain component."""

    def __new__(cls, *args, seed=0, components=None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(c, Component) for c in components]):
            return object.__new__(cls)
        return ContainerCS(Map.__name__, Map.__module__, components)

    def __init__(self, *args, seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        outerself = self

        class Enh(Enhancer):

            def _info_impl(self, data):
                return {"enhancer": map(lambda trf: trf.enhancer, outerself.component)}

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"stream": map(outerself.component.enhancer.transform, data.stream)}

        class Mod(Model):

            def _info_impl(self, train):
                return {"models": map(outerself.component.model, train.stream)}

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"stream": map(outerself.component.model(data).transform, data.stream)}

        super().__init__({}, Enh, Mod, seed, components, enhance, model, deterministic=True)
