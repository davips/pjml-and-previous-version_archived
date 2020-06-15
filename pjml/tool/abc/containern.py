from abc import ABC

from pjml.tool.abc.container import Container


class ContainerN(Container, ABC):
    """Container for more than one component."""

    def __init__(self, config, seed, components,
                 enhance=True, model=True,
                 deterministic=False):
        super().__init__(config, seed, components, enhance, model,
                         deterministic)

        self.size = len(components)
