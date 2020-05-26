from pjdata.infinitecollection import InfiniteCollection

from pjml.tool.abc.configless import LightConfigLess
from pjml.tool.model.model import Model


class Expand(LightConfigLess):
    def _apply_impl(self, data):
        applied = self._use_impl(data)
        return Model(self, data, applied)

    def _use_impl(self, data, **kwargs):
        transformation = self.transformations('u')[0]
        return InfiniteCollection(
            data,
            tuple(data.history) + tuple([transformation]),
            data.failure,
            data.uuid * transformation.uuid
        )
