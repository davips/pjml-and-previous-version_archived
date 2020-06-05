from typing import Any, Dict, Callable

from pjdata.types import Data


class DefaultModel:
    def _model_info(self, data: Data) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: Data) -> Callable[[Data], Data]:
        return lambda x: x
