from pjdata.infinitecollection import InfiniteCollection

from pjml.tool.abc.configless import LightConfigLess, TLightConfigLess
from pjml.tool.abc.mixin.component import TTransformer
from pjml.tool.model.model import Model


class Expand(LightConfigLess):
    def _apply_impl(self, data):
        applied = self._use_impl(data)
        return Model(self, data, applied)

    def _use_impl(self, data, **kwargs):
        transformation = self.transformations('u')[0]
        return InfiniteCollection(
            data,
            data.history + [transformation],
            data.failure,
            data.uuid00 + transformation.uuid00
        )


class TExpand(TLightConfigLess):
    def _modeler_impl(self, prior):
        return self._enhancer_impl()

    def _enhancer_impl(self):
        return TTransformer(func=self._func)

    def _func(self, data):
        transformation = self.transformations('u')[0]
        return InfiniteCollection(
            data,
            data.history + [transformation],
            data.failure,
            data.uuid00 + transformation.uuid00
        )

