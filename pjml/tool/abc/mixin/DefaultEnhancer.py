from typing import Any, Dict, Callable

from pjdata.aux.util import DataT


class DefaultEnhancer:
    def _enhancer_info(self, data: DataT) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[DataT], DataT]:
        return lambda data: data
