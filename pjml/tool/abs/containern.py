from abc import ABC

from pjml.tool.abs.container import Container


class ContainerN(Container, ABC):
    """Container for more than one component."""

    def __init__(self, config, seed, components, enhancer_cls, model_cls, enhance=True, model=True,
                 deterministic=False):
        super().__init__(config, seed, components, enhancer_cls, model_cls, enhance, model, deterministic)

        self.size = len(components)
