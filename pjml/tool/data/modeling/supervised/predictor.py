from abc import ABC

import pjdata.types as t
from pjdata.transformer.model import Model
from pjdata.transformer.pholder import PHolder
from pjml.tool.data.algorithm import SKLAlgorithm


class Predictor(SKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    def __init__(self, config, func, enhance, model):
        class PHo(PHolder):
            def _transform_impl(self, data):
                return data.frozen

        class Mod(Model):
            def _info_impl(self_, train):
                sklearn_model = self.algorithm_factory()
                sklearn_model.fit(*train.Xy())
                return {"sklearn_model": sklearn_model}

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"z": self.info.sklearn_model.predict(data.X)}

        super().__init__(config, func, enhancer_cls=PHo, model_cls=Mod, enhance=enhance, model=model)

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
