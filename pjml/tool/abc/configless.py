from abc import ABC

from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.mixin.component import Component


class TLightConfigLess(Component, ABC):
    """Parent class of all transformers without config. Also, apply==use."""

    def __init__(self, onenhancer=True, onmodel=True):
        super().__init__({}, onenhancer=onenhancer, onmodel=onmodel,
                         deterministic=True)

    @classmethod
    def _cs_impl(cls):
        return EmptyCS()

    def transformations(self, step, clean=True):
        return super().transformations('u')
