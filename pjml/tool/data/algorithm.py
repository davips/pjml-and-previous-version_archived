from abc import ABC
from functools import partial

from pjml.tool.abs.component import Component


class TSKLAlgorithm(Component, ABC):
    """Base class for scikitlearn algorithms."""

    def __init__(self, config, func, sklconfig=None, deterministic=False, **kwargs):
        Component.__init__(self, config, deterministic=deterministic, **kwargs)

        sklconfig = config if sklconfig is None else sklconfig

        if not deterministic:
            sklconfig = sklconfig.copy()

            # TODO: this won't be needed after defaults are enforced in all
            #  components.
            if "seed" not in sklconfig:
                sklconfig["seed"] = 0

            sklconfig["random_state"] = sklconfig.pop("seed")

        self.algorithm_factory = partial(func, **sklconfig)
