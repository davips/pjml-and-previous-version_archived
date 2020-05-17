from abc import ABC

from pjml.tool.abc.container1 import Container1
from pjml.tool.abc.containern import ContainerN, TContainerN


# TODO: Until now, every MinimalContainer is deterministic.
#  Every container propagates the seed to the config of its internal
#  transformers. So, it is determinist per se. However,
#  a MinimalContainer that is randomized in some way may appear in
#  the future.


class MinimalContainer1(Container1, ABC):
    """Container with minimum configuration (seed) for a single transformer.

    If more are given, they will be handled as a single Chain transformer."""

    def __init__(self, *args, seed=0, transformers=None):
        if transformers is None:
            transformers = args
        super().__init__({}, seed, transformers, deterministic=True)


class MinimalContainerN(ContainerN, ABC):
    """Container with minimum configuration (seed) for more than one
    transformer."""

    def __init__(self, *args, seed=0, transformers=None):
        if transformers is None:
            transformers = args
        super().__init__({}, seed, transformers, deterministic=True)


class TMinimalContainerN(TContainerN, ABC):
    """Container with minimum configuration (seed) for more than one
    transformer."""

    def __init__(self, *args, seed=0, transformers=None):
        if transformers is None:
            transformers = args
        super().__init__({}, seed, transformers, deterministic=True)

