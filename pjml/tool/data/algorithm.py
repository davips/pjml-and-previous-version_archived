from abc import ABC
from functools import partial

from pjml.tool.abc.mixin.component import TComponent


class TSKLAlgorithm(TComponent, ABC):
    """Base class for scikitlearn algorithms."""

    def __init__(self, config, func, sklconfig=None, deterministic=False,
                 **kwargs):
        TComponent.__init__(self, config, **kwargs,
                            deterministic=deterministic)

        sklconfig = config if sklconfig is None else sklconfig

        if not deterministic:
            sklconfig = sklconfig.copy()

            # TODO: this won't be needed after defaults are enforced in all
            #  components.
            if 'seed' not in sklconfig:
                sklconfig['seed'] = 0

            sklconfig['random_state'] = sklconfig.pop('seed')

        self.algorithm_factory = partial(func, **sklconfig)
