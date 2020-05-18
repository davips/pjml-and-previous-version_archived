from abc import ABC

from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.transformer import ISTransformer


class Invisible(ISTransformer, ABC):
    """Parent class of all atomic transformers that don't increase history
    of transformations.

    They are useful, but sometimes do not transform Data objects."""

    def transformations(self, step, clean=True):
        """Invisible components produce no transformations, so they need to
        override the list of expected transformations with []."""
        return []


class TInvisible(TComponent, ABC):
    """Parent class of all atomic transformers that don't increase history
    of transformations.

    They are useful, but sometimes do not transform Data objects."""

    def transformations(self, step, clean=True):
        """Invisible components produce no transformations, so they need to
        override the list of expected transformations with []."""
        return []

