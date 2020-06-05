from abc import ABC
from functools import lru_cache
from typing import Callable, Dict, Any, List

from pjdata.data import Data
from pjdata.step.transformation import Transformation
from pjml.tool.abc.mixin.DefaultEnhancer import DefaultEnhancer
from pjml.tool.abc.mixin.exceptionhandler import BadComponent
from pjml.tool.data.algorithm import TSKLAlgorithm


class TPredictor(DefaultEnhancer, TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    def _model_info(self, train: Data) -> Dict[str, Any]:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*train.Xy)
        return {"sklearn_model": sklearn_model}

    def _model_func(self, train: Data) -> Callable[[Data], Data]:
        info = self._model_info(train)

        def transform(posterior):  # old use
            return posterior.updated(
                self.transformations("u"), z=info["sklearn_model"].predict(posterior.X)
            )

        return transform

    @lru_cache()
    def transformations(self, step: str, clean: bool = True) -> List[Transformation]:
        if step == "a":
            return []
        elif step == "u":
            return [Transformation(self, step)]
        else:
            raise BadComponent("Wrong current step:", step)
