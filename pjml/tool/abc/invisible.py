from abc import ABC

from pjml.tool.abc.lighttransformer import LightTransformer


class Invisible(LightTransformer, ABC):
    """Parent class of all atomic transformers that don't increase history
    of transformations.

    They are useful, but sometimes do not transform Data objects."""

    def transformations(self, step, clean=True):
        """Invisible components produce no transformations, so they need to
        override the list of expected transformations with []."""
        return tuple()
