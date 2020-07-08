from abc import ABC
from functools import lru_cache
from typing import Callable, Any, Dict

import pjdata.types as t
from pjml.tool.abs.mixin.defaultenhancerimpl import withDefaultEnhancerImpl
from pjml.tool.data.algorithm import TSKLAlgorithm


class Predictor(withDefaultEnhancerImpl, TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*data.Xy())
        return {"sklearn_model": sklearn_model}

    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Result]:
        info = self._model_info(data)
        return lambda d: {'z': info["sklearn_model"].predict(d.X)}

    def _enhancer_func(self) -> Callable[[t.Data], t.Result]:
        return lambda posterior: posterior.frozen
