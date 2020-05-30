from functools import lru_cache

from numpy.random.mtrand import uniform
from sklearn.decomposition import PCA as SKLPCA

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import RealP, FixedP
from pjml.tool.abc.mixin.transformer import TTransformer
from pjml.tool.data.algorithm import TSKLAlgorithm


class PCA(TSKLAlgorithm):
    def __init__(self, onenhancer=True, onmodel=True, **sklconfig):
        super().__init__(sklconfig, SKLPCA, onenhancer=onenhancer,
                         onmodel=onmodel)

    @lru_cache()
    def _info(self, prior):  # old apply
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(prior.X)
        return {'sklearn_model': sklearn_model}

    def predict(self, prior, posterior):  # old use
        info = self._info(prior)
        return posterior.updated(
            self.transformations('u'),  # desnecessário?
            X=info['sklearn_model'].transform(posterior.X)
        )

    def _model_impl(self, prior):
        return TTransformer(
            func=lambda posterior: self.predict(prior, posterior),
            info=self._info(prior)
        )

    def _enhancer_impl(self):
        return TTransformer(
            func=lambda prior: self.predict(prior, prior),
            info=self._info
        )

    @classmethod
    def _cs_impl(cls):
        # todo: set random seed; set 'cache_size'
        param = {
            'n_components': RealP(uniform, low=0.0, high=1.0),
            'copy': FixedP(True),
            'whiten': FixedP(False),
            'svd_solver': FixedP('auto'),
            'tol': FixedP(0.0),
            'iterated_power': FixedP('auto'),
        }
        return TransformerCS(nodes=Node(param))
