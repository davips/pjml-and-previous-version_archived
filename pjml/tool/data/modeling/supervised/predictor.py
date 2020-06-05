from abc import ABC
from typing import Callable, Any, Dict

import pjdata.types as t
from pjdata.transformer import Transformer
from pjml.tool.abc.mixin.defaultenhancer import DefaultEnhancer
from pjml.tool.data.algorithm import TSKLAlgorithm


class TPredictor(DefaultEnhancer, TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    def _model_info(self, train: t.Data) -> Dict[str, Any]:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*train.Xy)
        return {"sklearn_model": sklearn_model}

    def _model_func(self, train: t.Data) -> Callable[[t.Data], t.Data]:
        info = self._model_info(train)

        def transform(posterior):  # old use
            return posterior.updated(
                (), z=info["sklearn_model"].predict(posterior.X)
            )

        return transform

    def _enhancer_impl(self):
        return Transformer(func=lambda posterior: posterior.frozen, info={})
