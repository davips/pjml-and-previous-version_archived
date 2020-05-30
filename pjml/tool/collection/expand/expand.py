from itertools import repeat

from pjdata.collection import Collection
from pjml.tool.abc.configless import TLightConfigLess
from pjml.tool.abc.mixin.transformer import TTransformer


class Expand(TLightConfigLess):
    def dual_transform(self, prior, posterior):
        print(self.__class__.__name__, ' dual transf (((')
        if self.onenhancer:
            prior = Collection(repeat(prior), lambda: prior, finite=False,
                               debug_info='expand')
        if self.onmodel:
            posterior = Collection(repeat(posterior), lambda: posterior,
                                   finite=False, debug_info='expand')
        return prior, posterior

    def _model_impl(self, prior):
        return self._enhancer_impl()

    def _enhancer_impl(self):
        def transform(data):
            iterator = repeat(data)
            return Collection(iterator, lambda: data, finite=False,
                              debug_info='expand')

        return TTransformer(
            func=transform,
            info=None
        )
