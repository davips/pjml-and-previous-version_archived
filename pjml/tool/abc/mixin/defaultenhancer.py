from typing import Any, Dict, Callable

from pjdata.content.data import Data


class DefaultEnhancer:
    def _enhancer_info(self, data: Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[Data], Data]:
        return lambda data: data
