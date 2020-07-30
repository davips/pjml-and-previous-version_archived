from abc import abstractmethod

import pjml.tool.abs.component as co


class Macro(co.Component):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, enhancer_cls=self.component.enhancer_cls, model_cls=self.component.model_cls, **kwargs)

    @property
    @abstractmethod
    def component(self) -> co.Component:
        pass

    def _cs_impl(self):
        return self.component.cs

    def _cfuuid_impl(self, data=None):
        """Macro is special case, and needs to calculate the uuid based on its internal component."""
        return self.component.cfuuid(data)
