from itertools import repeat

from pjdata.infinitecollection import InfiniteCollection

from pjml.tool.abc.configless import LightConfigLess, TLightConfigLess
from pjml.tool.abc.mixin.component import TTransformer
from pjml.tool.model.model import Model


class TExpand(TLightConfigLess):
    def _model_impl(self, prior):
        return self._enhancer_impl()

    def _enhancer_impl(self):
        def transform(data):
            return repeat(data)

        return TTransformer(
            func=transform,
            info=None
        )
