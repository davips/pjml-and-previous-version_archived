from abc import ABC

import pjdata.types as t
from pjdata.transformer.model import Model
from pjdata.transformer.pholder import PHolder
from pjdata.transformer.transformer import Transformer
from pjml.tool.data.algorithm import SKLAlgorithm


class Predictor(SKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    def _enhancer_impl(self) -> Transformer:
        return PHolder(self, lambda data: data.frozen)

    def _model_impl(self, data: t.Data) -> Model:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*data.Xy())
        return Model(self, lambda test: {"z": sklearn_model.predict(test.X)}, {"sklearn_model": sklearn_model}, data)


    # def _model_impl(self, data: t.Data) -> Model:
    #     def lazymodel():
    #         sklearn_model = self.algorithm_factory()
    #         sklearn_model.fit(*data.Xy())
    #         return sklearn_model
    #
    #     return Model(self,
    #                  func=lambda info, test: {"z": info.sklearn_model.predict(test.X)},
    #                  info={"sklearn_model": lazymodel},
    #                  data=data)
