from typing import Tuple, Iterator

from itertools import repeat

from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.nonfinalizer import NonFinalizer
from itertools import tee, repeat
from typing import Union, Tuple, Optional

from pjdata.collection import Collection
from pjdata.data import Data
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.util import TDatas, TDatasTuple


class Reduce(Invisible, NonFinalizer, Component):
    def __init__(self, config: Optional[dict]=None, **kwargs):
        # TODO: delete onenhance/onmodel? se não consumir pode explodir
        config = {} if config is None else config
        super().__init__(config, deterministic=True, **kwargs)

    def enhancer_info(self):
        pass

    def model_info(self, data):
        pass

    def enhancer_func(self) -> Transformer:
        def transform(collection: Collection) -> Collection:
            # Exhaust iterator.
            c = 0
            print('\nReduce starts loop... >>>>>>>>>>>>>>>>>>>>>>>>>>>')
            for d in collection:
                print('  Reduce consumed item', c, '\n')
                c += 1
                pass
            print('...Reduce exits loop. <<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
            print('  Reduce asks for pendurado at...', collection.debug_info)
            return collection.data  # type: ignore

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

    def dual_transform(self, train_collection, test_collection) -> Union[Tuple[Data, Data], Tuple[Data, Tuple[Data, ...]]]:
        # # Handle non-collection cases.  <- makes no sense
        # if not self.onenhancer and not self.onmodel:
        #     return train_collection, test_collection
        # train_iterator = train_collection if self.onenhancer else repeat(None)
        # train_iterator = train_collection if self.onenhancer else repeat(None)

        # Consume iterators.
        for _ in zip(train_collection, test_collection):
            pass

        return train_collection.data, test_collection.data


        # As @property is not recognized, mypy raises an error saying that this
        # property coll.data does not exist.
