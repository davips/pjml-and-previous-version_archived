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

        outerself = self

        class Mod(Model):
            def _info_impl(self, train):
                sklearn_model = outerself.algorithm_factory()
                sklearn_model.fit(*train.Xy())
                return {"sklearn_model": sklearn_model}

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"z": self.info.sklearn_model.predict(data.X)}

        super().__init__(config, func, enhancer_cls=PHo, model_cls=Mod, enhance=enhance, model=model)
