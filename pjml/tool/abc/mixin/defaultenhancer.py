from functools import lru_cache
from typing import Any, Dict, Callable

from pjdata.types import DataOrColl


class DefaultEnhancer:
    @lru_cache()
    def _enhancer_info(self, data: DataOrColl) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[DataOrColl], DataOrColl]:
        return lambda data: data
