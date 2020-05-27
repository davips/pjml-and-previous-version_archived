from itertools import repeat

from pjdata.collection import Collection
from pjml.tool.abc.configless import TLightConfigLess
from pjml.tool.abc.mixin.component import TTransformer


class TExpand(TLightConfigLess):
    def _model_impl(self, prior):
        return self._enhancer_impl()

    def _enhancer_impl(self):
        def transform(data):
            generator = repeat(data)
            return Collection(generator, lambda: data, finite=False,debug_info='expand')

        return TTransformer(
            func=transform,
            info=None
        )
