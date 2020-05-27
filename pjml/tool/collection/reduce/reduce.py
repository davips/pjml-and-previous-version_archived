from pjml.tool.abc.invisible import TInvisible
from pjml.tool.abc.mixin.component import TTransformer


# class TReduce(TComponent, FunctionInspector, ABC):
#     def __init__(self, config, deterministic=False, **kwargs):
#         super().__init__(config, deterministic, **kwargs)
#         self.function = self.function_from_name[config['function']]
#
#
#     def transformations(self, step, clean=True):
#         return super().transformations('u')


class TRReduce(TInvisible):
    @classmethod
    def _cs_impl(cls):
        pass

    def __init__(self, config=None, deterministic=True, **kwargs):
        config = {} if config is None else config
        super().__init__(config, deterministic, **kwargs)

    def _enhancer_impl(self):
        def transform(collection):
            # Exhaust iterator.
            c = 0
            print('\nReduce asks to consume item', c)
            for d in collection:
                c += 1
                print('\nReduce asks to consume item', c)
                pass
            return collection.data

        return TTransformer(
            func=transform,
            info=None
        )

    def _model_impl(self, prior):
        return self._enhancer_impl()
