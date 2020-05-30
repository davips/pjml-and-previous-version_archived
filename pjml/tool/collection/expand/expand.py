from itertools import repeat

from pjdata.collection import Collection
from pjdata.data import Data
from pjml.tool.abc.configless import ConfigLess
from pjml.tool.abc.mixin.transformer import Transformer


class Expand(ConfigLess):
    def dual_transform(self, prior: Data, posterior: Data):
        print(self.__class__.__name__, ' dual transf (((')
        if self.onenhancer:
            prior = Collection(repeat(prior), lambda: prior, finite=False,
                               debug_info='expand')
        if self.onmodel:
            posterior = Collection(repeat(posterior), lambda: posterior,
                                   finite=False, debug_info='expand')
        return prior, posterior

    def _model_impl(self, prior: Data) -> Transformer:
        return self._enhancer_impl()

    def _enhancer_impl(self) -> Transformer:
        def transform(data):
            iterator = repeat(data)
            return Collection(iterator, lambda: data, finite=False,
                              debug_info='expand')

        return Transformer(
            func=transform,
            info=None
        )
