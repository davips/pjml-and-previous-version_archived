from functools import lru_cache

from numpy.random.mtrand import uniform
from sklearn.decomposition import PCA

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import RealP, FixedP
from pjml.tool.abc.mixin.component import TTransformer
from pjml.tool.data.algorithm import TSKLAlgorithm


class TPCA(TSKLAlgorithm):
    def __init__(self, **sklconfig):
        super().__init__(sklconfig, PCA)

    @lru_cache()
    def _info(self, prior):  # old apply
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(prior.X)
        return {'sklearn_model': sklearn_model}

    def _modeler_impl(self, prior):
        info = self._info(prior)

        def predict(posterior):  # old use
            return posterior.updated(
                self.transformations('u'),  # desnecessário?
                X=info['sklearn_model'].transform(posterior.X)
            )
        return TTransformer(func=predict, **info)

    def _enhancer_impl(self):
        def predict(prior):  # old use
            info = self._info(prior)
            return prior.updated(
                self.transformations('u'),  # desnecessário?
                X=info['sklearn_model'].transform(prior.X)
            )
        return TTransformer(func=predict)

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
