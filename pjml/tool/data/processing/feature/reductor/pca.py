from functools import lru_cache

from numpy.random.mtrand import uniform
from sklearn.decomposition import PCA as SKLPCA

import pjdata.types as t
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import RealP, FixedP
from pjml.tool.data.algorithm import SKLAlgorithm


class PCA(SKLAlgorithm):
    # TODO:
    #  Adopt explicit parameters in all components
    #  Reason:
    #   better for auto-completion of docs, webdocs, parsing/refactoring, IDE tips, command line/ipython, notebooks,
    #   definition of default values etc.
    #       (regarding def. values: we can easily get it through PCA().config
    #           at zero cost since algorithm_factory is not actually called at init)
    # TODO:
    #  adopt sensible simple and common names for parameters
    #  to allow a homogeneous "pajé-style" interface across different ML libraries.
    #  Example: In the PCA context 'n' is obviously the number of features.
    def __init__(self, n: int = 2, enhance: bool = True, model: bool = True):
        outerself = self

        class Enh(Enhancer):
            def _info_impl(self, data):
                return outerself._info(data)

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"X": self.info.sklearn_model.transform(data.X)}

        class Mod(Model):
            def _info_impl(self, train):
                return outerself._info(train)

            def _transform_impl(self, data: t.Data) -> t.Result:
                return {"X": self.info.sklearn_model.transform(data.X)}

        super().__init__(
            {"n": n}, SKLPCA, Enh, Mod, deterministic=True, sklconfig={"n_components": n}, enhance=enhance, model=model
        )

        # INTERESTING: PCA has 2 UUIDs

    @lru_cache()
    def _info(self, train):
        sklearn_model = self.algorithm_factory()
        sklearn_model.fit(train.X)
        return {"sklearn_model": sklearn_model}

    @classmethod
    def _cs_impl(cls) -> CS:
        # todo: set random seed; set 'cache_size'
        param = {
            "n": RealP(uniform, low=0.0, high=1.0),
            "copy": FixedP(True),
            "whiten": FixedP(False),
            "svd_solver": FixedP("auto"),
            "tol": FixedP(0.0),
            "iterated_power": FixedP("auto"),
        }
        return CS(nodes=[Node(param)])
