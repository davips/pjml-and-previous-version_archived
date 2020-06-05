from functools import lru_cache
from typing import Any, Dict, Callable

from pjdata.types import DataOrColl


class DefaultModel:
    @lru_cache()
    def _model_info(self, data: DataOrColl) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: DataOrColl) -> Callable[[DataOrColl], DataOrColl]:
        return lambda x: x
