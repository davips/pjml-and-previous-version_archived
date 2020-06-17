from typing import Any, Dict, Callable

from pjdata.types import Data


class DefaultEnhancer:
    def _enhancer_info(self, data: Data = None) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[Data], Data]:
        return lambda data: data
