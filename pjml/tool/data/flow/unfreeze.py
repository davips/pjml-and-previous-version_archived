from typing import Callable, Dict, Any

import pjdata.types as t
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component


class Unfreeze(Component):
    """Resurrect a workflow by unfreezing a Data object.
    Useful when the workflow contains, e.g. more than one classifier.
    """

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {}

    def _model_info(self, test: NoData) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        return lambda data: data.unfrozen

    def _model_func(self, train: t.Data) -> Callable[[t.Data], t.Data]:
        return lambda data: data.unfrozen

    @classmethod
    def _cs_impl(cls) -> ConfigSpace:
        return EmptyCS()
