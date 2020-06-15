from functools import lru_cache
from typing import List

from pjdata.transformer import Transformer


class Invisible:
    """Parent class of all atomic components that don't increase history
    of transformations.

    They are useful, but sometimes do not transform Data objects."""

    @lru_cache()
    def transformations(
            self,
            step: str,
            clean: bool = True
    ) -> List[Transformer]:

        """Invisible components produce no transformations, so they need to
        override the list of expected transformations with []."""
        return []
