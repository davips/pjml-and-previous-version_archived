from abc import abstractmethod, ABC
from typing import Any

import pjdata.types as t
import pjml.tool.abc.mixin.component as co
from pjdata.types import Data


class Macro(ABC):
    @property
    @abstractmethod
    def component(self) -> co.Component:
        pass

    def _enhancer_info(self, data: t.Data) -> t.Dict[str, Any]:
        return self.component.enhancer.info(data)

    def _model_info(self, data: Data) -> t.Dict[str, Any]:
        return self.component.model(data).info

    def _enhancer_func(self) -> t.Transformation:
        return self.component.enhancer.transform

    def _model_func(self, data: Data) -> t.Transformation:
        return self.component.model(data).transform
