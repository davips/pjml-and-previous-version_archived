from abc import ABC

from pjml.tool.abs.container1 import Container1

# TODO: Until now, every MinimalContainer is deterministic.
#  Every container propagates the seed to the config of its internal
#  components. So, it is determinist per se. However,
#  a MinimalContainer that is randomized in some way may appear in
#  the future.
from pjml.tool.abs.containern import ContainerN


class MinimalContainer1(Container1, ABC):
    """Container with minimum configuration (seed) for a single component.

    If more are given, they will be handled as a single Chain component."""

    def __init__(self, *args, seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        super().__init__({}, seed, components, enhance, model, deterministic=True)


class MinimalContainerN(ContainerN, ABC):
    """Container with minimum configuration (seed) for more than one component."""

    def __init__(self, *args, seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        super().__init__({}, seed, components, enhance, model, deterministic=True)
