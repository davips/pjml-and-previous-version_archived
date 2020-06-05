from abc import ABC
from functools import lru_cache

from pjdata.step.transformation import Transformation
from pjdata.step.transformer import Transformer
from pjml.tool.abc.mixin.exceptionhandler import BadComponent
from pjml.tool.data.algorithm import TSKLAlgorithm


class TPredictor(TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    @lru_cache()
    def _info(self, prior):  # old apply
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*prior.Xy)
        return {'sklearn_model': sklearn_model}

    def _model_impl(self, prior):
        info = self._info(prior)

        def transform(posterior):  # old use
            return posterior.updated(
                self.transformations('u'),
                # desnecessário ? #TODO: memory leak?
                z=self._info(prior)['sklearn_model'].predict(posterior.X)
            )

        return Transformer(func=transform, info=info)

    def _enhancer_impl(self):
        return Transformer(func=lambda posterior: posterior.frozen, info={})

    def transformations(self, step, clean=True):
        if step == 'a':
            return []
        elif step == 'u':
            return [Transformation(self, step)]
        else:
            raise BadComponent('Wrong current step:', step)
