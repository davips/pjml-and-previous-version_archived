from abc import ABC
from functools import partial

from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling
from pjml.tool.abs.mixin.noinfo import withNoInfo


class SKLAlgorithm(Component, ABC):
    """Base class for scikitlearn algorithms."""

    def __init__(self, config, func, enhancer_cls, model_cls, sklconfig=None, deterministic=False, **kwargs):
        super().__init__(config, enhancer_cls=enhancer_cls, model_cls=model_cls, deterministic=deterministic, **kwargs)

        sklconfig = config if sklconfig is None else sklconfig

        if not deterministic:
            sklconfig = sklconfig.copy()

            # TODO: this won't be needed after defaults are enforced in all
            #  components.
            if "seed" not in sklconfig:
                sklconfig["seed"] = 0

            sklconfig["random_state"] = sklconfig.pop("seed")

        self.algorithm_factory = partial(func, **sklconfig)
