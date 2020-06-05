from typing import Union, Tuple, Optional, Any, Dict, Callable

from pjdata.aux.util import Property
from pjdata.content.collection import Collection
from pjdata.content.data import Data
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.nonfinalizer import NonFinalizer


class Reduce(Invisible, NonFinalizer, Component):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        # TODO: delete onenhance/onmodel? se não consumir pode explodir
        config = {} if config is None else config
        super().__init__(config, deterministic=True, **kwargs)

    def _enhancer_info(self, train_coll: Collection) -> Dict[str, Any]:
        return {}

    def _model_info(self, train_coll) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[Collection], Data]:
        def transform(test_coll: Collection) -> Data:
            # Exhaust iterator.
            c = 0
            print('\nReduce starts loop... >>>>>>>>>>>>>>>>>>>>>>>>>>>')
            for d in test_coll:
                print('  Reduce consumed item', c, '\n')
                c += 1
                pass
            print('...Reduce exits loop. <<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
            print('  Reduce asks for pendurado at...', test_coll.debug_info)
            return test_coll.data

        return transform

    def _model_func(self, train_coll: Collection) -> Callable[[Collection], Data]:
        return self._enhancer_func()

    @Property
    def finite(self) -> bool:
        return False

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError

    def dual_transform(
            self,
            train_collection: Collection,
            test_collection: Collection
    ) -> Union[Tuple[Data, Data], Tuple[Data, Tuple[Data, ...]]]:
        # # Handle non-collection cases.  <- makes no sense
        # if not self.enhance and not self.onmodel:
        #     return train_collection, test_collection
        # train_iterator = train_collection if self.enhance else repeat(None)
        # train_iterator = train_collection if self.enhance else repeat(None)

        # Consume iterators.
        for _ in zip(train_collection, test_collection):
            pass

        return train_collection.data, test_collection.data

        # As @property is not recognized, mypy raises an error saying that this
        # property coll.data does not exist.
