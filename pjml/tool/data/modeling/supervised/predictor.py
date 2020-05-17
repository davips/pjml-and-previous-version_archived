from abc import ABC
from functools import lru_cache

from pjdata.step.transformation import Transformation
from pjml.tool.abc.mixin.component import TTransformer
from pjml.tool.abc.mixin.enforceapply import EnforceApply
from pjml.tool.abc.mixin.exceptionhandler import BadComponent
from pjml.tool.data.algorithm import HeavyAlgorithm, TSKLAlgorithm
from pjml.tool.model.model import Model


class Predictor(HeavyAlgorithm, EnforceApply, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """

    def _apply_impl(self, data):
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*data.Xy)
        return Model(self, data, data.frozen, sklearn_model=sklearn_model)

    def _use_impl(self, data, sklearn_model=None):
        return data.updated(
            self.transformations('u'),
            z=sklearn_model.predict(data.X)
        )

    def transformations(self, step, clean=True):
        if step == 'a':
            return []
        elif step == 'u':
            return [Transformation(self, step)]
        else:
            raise BadComponent('Wrong current step:', step)


class TPredictor(TSKLAlgorithm, ABC):
    """
    Base class for classifiers, regressors, ... that implement fit/predict.
    """
    @lru_cache()
    def _info(self, prior):  # old apply
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(*prior.Xy)
        return {'sklearn_model': sklearn_model}

    def _modeler_impl(self, prior):
        info = self._info(prior)

        def predict(posterior):  # old use
            return posterior.updated(
                self.transformations('u'),  # desnecessário
                z=self._info(prior)['sklearn_model'].predict(posterior.X)
            )
        return TTransformer(func=predict, **info)

    def transformations(self, step, clean=True):
        if step == 'a':
            return []
        elif step == 'u':
            return [Transformation(self, step)]
        else:
            raise BadComponent('Wrong current step:', step)

