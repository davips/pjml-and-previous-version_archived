from abc import ABC

from pjml.tool.abc.container import Container, TContainer


class ContainerN(Container, ABC):
    """Container for more than one transformer."""

    def __init__(self, config, seed, transformers, deterministic):
        super().__init__(config, seed, transformers, deterministic)

        self.size = len(transformers)


class TContainerN(TContainer, ABC):
    """Container for more than one transformer."""

    def __init__(self, config, seed, transformers, deterministic):
        super().__init__(config, seed, transformers, deterministic)

        self.size = len(transformers)
