from typing import Any, Dict, Callable

from pjdata.aux.util import DataT


class DefaultModel:
    def _model_info(self, data: DataT) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: DataT) -> Callable[[DataT], DataT]:
        return lambda x: x
