from abc import abstractmethod
from functools import lru_cache
from typing import Any, Dict, Callable

from pjdata.types import Data


class DefaultModel:
    # @property
    # @abstractmethod
    # def name(self):
    #     pass

    @lru_cache()
    def _model_info(self, data: Data) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: Data) -> Callable[[Data], Data]:
        # print(self.name)
        return lambda x: x
