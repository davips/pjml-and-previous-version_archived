from typing import Dict, Any, Callable

from numpy.random.mtrand import uniform
from sklearn.decomposition import PCA as SKLPCA

from pjdata.types import Data
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import RealP, FixedP
from pjml.tool.data.algorithm import TSKLAlgorithm


class PCA(TSKLAlgorithm):
    def __init__(self, enhance=True, model=True, **sklconfig):
        super().__init__(sklconfig, SKLPCA, enhance=enhance,
                         model=model)
    def __init__(self, enhancer: bool = True, model: bool = True, **kwargs):
        super().__init__(kwargs, SKLPCA, enhancer=enhancer, model=model)

    def _enhancer_info(self, data: Data) -> Dict[str, Any]:
        return self._info(data)

    def _enhancer_func(self) -> Callable[[Data], Data]:
        return lambda train: self.predict(train, train)

    def _model_info(self, train: Data) -> Dict[str, Any]:
        return self._info(train)

    def _model_func(self, train: Data) -> Callable[[Data], Data]:
        return lambda test: self.predict(train, test)

    def _info(self, data: Data) -> Dict[str, Any]:
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(data.X)
        return {"sklearn_model": sklearn_model}

    def predict(self, train: Data, test: Data) -> Data:
        info = self._info(train)
        return test.updated(
            self.transformations("u"),  # desnecessário?
            X=info["sklearn_model"].transform(test.X),
        )

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        # todo: set random seed; set 'cache_size'
        param = {
            "n_components": RealP(uniform, low=0.0, high=1.0),
            "copy": FixedP(True),
            "whiten": FixedP(False),
            "svd_solver": FixedP("auto"),
            "tol": FixedP(0.0),
            "iterated_power": FixedP("auto"),
        }
        return TransformerCS(nodes=Node(param))
