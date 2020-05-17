from abc import ABC

from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.transformer import INSTransformer, DTransformer


class HeavyConfigLess(DTransformer, ABC):
    """Parent class of all transformers without config."""

    def __init__(self):
        super().__init__({}, deterministic=True)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()


class LightConfigLess(INSTransformer, ABC):
    """Parent class of all transformers without config. Also, apply==use."""

    def __init__(self):
        super().__init__({}, deterministic=True)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()

    def transformations(self, step, clean=True):
        return super().transformations('u')


class TLightConfigLess(TComponent, ABC):
    """Parent class of all transformers without config. Also, apply==use."""

    def __init__(self):
        super().__init__({}, deterministic=True)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()

    def transformations(self, step, clean=True):
        return super().transformations('u')
