from abc import ABC

from pjdata.specialdata import NoData
from pjml.tool.abc.invisible import TInvisible
from pjml.tool.abc.mixin.component import TComponent, TTransformer
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


class TReduce(TComponent, FunctionInspector, ABC):
    def __init__(self, config, deterministic=False, **kwargs):
        super().__init__(config, deterministic, **kwargs)
        self.function = self.function_from_name[config['function']]

    # This is not necessary because it is done in the father class
    # def _apply_impl(self, collection):
    #     applied = self._use_impl(collection)
    #     return Model(self, collection, applied)

    def transformations(self, step, clean=True):
        return super().transformations('u')


class TRReduce(TInvisible):
    @classmethod
    def _cs_impl(cls):
        pass

    def __init__(self, config=None, deterministic=True, **kwargs):
        config = {} if config is None else config
        super().__init__(config, deterministic, **kwargs)

    def _enhancer_impl(self, step='e'):
        def func(collection):
            if collection.has_nones:
                raise Exception(
                    "Warning: You shuld use 'Shrink()' to handling collections "
                    "with None. ")

            res = collection.original_data.matrices.copy()
            res.update(collection.fields)
            return NoData.updated(
                collection.history,
                failure=collection.failure,
                **res
            )

        return TTransformer(
            func=func,
            info=None
        )

    def _model_impl(self, prior, step='m'):
        return self._enhancer_impl(step)
