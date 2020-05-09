from abc import ABC

from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import ISTransformer


class Reduce(ISTransformer, FunctionInspector, ABC):
    def __init__(self, config, deterministic=False):
        super().__init__(config, deterministic)
        self.function = self.function_from_name[config['function']]

    # This is not necessary because it is done in the father class
    # def _apply_impl(self, collection):
    #     applied = self._use_impl(collection)
    #     return Model(self, collection, applied)

    def transformations(self, step, clean=True):
        return super().transformations('u')
