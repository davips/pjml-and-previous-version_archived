from abc import ABC
from functools import lru_cache
from typing import Callable, Any, Dict

import pjdata.types as t
from pjdata.transformer import Transformer
from pjml.tool.abc.mixin.exceptionhandler import BadComponent
from pjml.tool.data.algorithm import TSKLAlgorithm


class TPredictor(TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    @lru_cache()
    def _enhancer_info(self, data: t.Data) -> Dict[str, Any]:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*data.Xy)
        return {'sklearn_model': sklearn_model}

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {}

    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Data]:
        def transform(posterior):
            return posterior.updated(
                (), z=self._enhancer_info(data)['sklearn_model'].predict(posterior.X)
            )

        return transform

    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        return lambda posterior: posterior.frozen
