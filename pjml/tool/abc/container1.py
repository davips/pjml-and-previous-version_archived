from abc import ABC

from pjml.tool.abc.container import Container


class Container1(Container, ABC):
    """Configurable container for a single component.

    If more are given, they will be handled as a single Seq component."""

    def __init__(self, config, seed, components, enhance, model,
                 deterministic):
        super().__init__(config, seed, components, enhance, model,
                         deterministic)

        # Implementation-wise, Container1(Chain(a,b,c)) is needed to make
        # Container1(a,b,c) possible.
        if len(self.components) > 1:
            from pjml.tool.chain import Chain
            self.component = Chain(components=self.components)
        else:
            self.component = self.components[0]
