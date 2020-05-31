from typing import Tuple, Iterator

from itertools import repeat

from pjml.tool.abc.invisible import TInvisible
from pjml.tool.abc.nonfinalizer import NonFinalizer


class TRReduce(TInvisible, NonFinalizer):
    def __init__(self, config=None, **kwargs):
        # TODO: delete onenhance/onmodel? se não consumir pode explodir
        config = {} if config is None else config
        super().__init__(config, **kwargs, deterministic=True)

    def enhancer_info(self):
        pass

    def model_info(self, data):
        pass

    def enhancer_func(self):
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

        return transform

    def model_func(self, data):
        return self.enhancer_func()

    @property
    def finite(self):
        return False

    # def iterators(self, train, test) -> Tuple[Iterator]:
    #     pass

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError

    def dual_transform(self, train_collection, test_collection):
        # # Handle non-collection cases.  <- makes no sense
        # if not self.onenhancer and not self.onmodel:
        #     return train_collection, test_collection
        # train_iterator = train_collection if self.onenhancer else repeat(None)
        # train_iterator = train_collection if self.onenhancer else repeat(None)

        # Consume iterators.
        for _ in zip(train_collection, test_collection):
            pass

        return train_collection.data, test_collection.data
