from abc import ABC

from pjml.tool.abc.container import Container


class Container1(Container, ABC):
    """Configurable container for a single transformer.

    If more are given, they will be handled as a single Seq transformer."""

    def __init__(self, config, seed, transformers, onenhancer, onmodel,
                 deterministic):
        super().__init__(config, seed, transformers, onenhancer, onmodel,
                         deterministic)

        # Implementation-wise, Container1(Chain(a,b,c)) is needed to make
        # Container1(a,b,c) possible.
        if len(self.transformers) > 1:
            from pjml.tool.chain import Chain
            self.transformer = Chain(transformers=self.transformers)
        else:
            self.transformer = self.transformers[0]
