from abc import ABC
from functools import partial

from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.tool.abs.component import Component


class SKLAlgorithm(Component, ABC):
    """Base class for scikitlearn algorithms."""

    def __init__(self, config, func, sklconfig=None, deterministic=False, **kwargs):
        # class MLPEnhancer(Enhancer):
        #     def transform(data):  # _transformer_impl?
        #         return data.frozen
        #
        # class MLPModel(Model):
        #     def info(self, data):
        #         return Multilayer(cfg1, cfg2).fit(*data.Xy)
        #
        #     def transform(data):  # _transformer_impl?
        #         y = self.info.skmodel.predict(data.X)
        #         return data.updated(self, y=y)

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
