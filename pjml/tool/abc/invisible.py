from functools import lru_cache
from typing import List

from pjdata.step.transformation import Transformation


class Invisible:
    """Parent class of all atomic transformers that don't increase history
    of transformations.

    They are useful, but sometimes do not transform Data objects."""

    @lru_cache()
    def transformations(
            self,
            step: str,
            clean: bool = True
    ) -> List[Transformation]:

        """Invisible components produce no transformations, so they need to
        override the list of expected transformations with []."""
        return []
