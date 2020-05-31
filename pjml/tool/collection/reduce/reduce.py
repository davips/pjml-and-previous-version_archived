from itertools import tee, repeat

from pjml.tool.abc.invisible import TInvisible


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

    def __init__(self, config=None, **kwargs):
        config = {} if config is None else config
        super().__init__(config, **kwargs, deterministic=True)

    def dual_transform(self, prior_collection, posterior_collection):
        print(self.__class__.__name__, ' dual transf (((')

        # Handle non collection cases.
        if not self.onenhancer and not self.onmodel:
            return prior_collection, posterior_collection
        if not self.onenhancer:
            prior_collection = repeat(None)
        if not self.onmodel:
            posterior_collection = repeat(None)

        # Consume iterators.
        for _ in zip(prior_collection, posterior_collection):
            pass

        return prior_collection.data, posterior_collection.data

    def _enhancer_impl(self):
        def transform(collection):
            # Exhaust iterator.
            c = 0
            print('\nReduce starts loop... >>>>>>>>>>>>>>>>>>>>>>>>>>>')
            for d in collection:
                print('  Reduce consumed item', c, '\n')
                c += 1
                pass
            print('...Reduce exits loop. <<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
            print('  Reduce asks for pendurado at...', collection.debug_info)
            return collection.data

        return TTransformer(
            func=transform,
            info=None
        )

    def _model_impl(self, prior):
        return self._enhancer_impl()
