from abc import abstractmethod, ABC
from typing import Any

import pjdata.types as t
import pjml.tool.abs.component as co
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.transformer import Transformer
from pjdata.types import Data


class Macro(co.Component):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, enhancer_cls=self.component.enhancer_cls, model_cls=self.component.model_cls, **kwargs)

    @property
    @abstractmethod
    def component(self) -> co.Component:
        pass

    def _cs_impl(self):
        return self.component.cs
