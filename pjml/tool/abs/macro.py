from abc import abstractmethod, ABC
from typing import Any

import pjdata.types as t
import pjml.tool.abs.component as co
from pjdata.transformer.transformer import Transformer
from pjdata.types import Data


class Macro(ABC):
    @property
    @abstractmethod
    def component(self) -> co.Component:
        pass

    def _enhancer_impl(self) -> Transformer:
        return self.component.enhancer

    def _model_impl(self, data: t.Data) -> Transformer:
        return self.component.model(data)

    def _cs_impl(self):
        return self.component.cs
