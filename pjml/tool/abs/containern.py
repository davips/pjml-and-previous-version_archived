from abc import ABC

from pjml.tool.abs.container import Container


class ContainerN(Container, ABC):
    """Container for more than one component."""

    def __init__(
            self, config, enhancer_cls, model_cls, seed, components, enhance=True, model=True, deterministic=False
    ):
        super().__init__(config, enhancer_cls, model_cls, seed, components, enhance, model, deterministic)

        self.size = len(components)
