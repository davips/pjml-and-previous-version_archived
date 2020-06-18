from functools import lru_cache
from typing import Union, Tuple, Optional, Any, Dict, Callable

import pjdata.types as t
from pjdata.content.data import Data
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.component import Component


class Reduce(Invisible, Component):
    def __init__(self, config: dict = None, **kwargs):
        # TODO: delete onenhance/onmodel? senão, consumir pode explodir
        config = {} if config is None else config
        super().__init__(config, deterministic=True, **kwargs)

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {}

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> t.Transformation:
        def transform(data: Data) -> t.Result:
            # Exhaust iterator.
            c = 0
            print('\nReduce starts loop... >>>>>>>>>>>>>>>>>>>>>>>>>>>')
            for d in data.stream:
                print('  Reduce consumed item', c, '\n')
                c += 1
                pass
            print('...Reduce exits loop. <<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
            return {'stream': None}

        return transform

    def _model_func(self, data: t.Data) -> t.Transformation:
        return self._enhancer_func()

    @classmethod
    def _cs_impl(cls):
        params={}
        return CS(nodes=[Node(params)])

    def dual_transform(self, train: t.Data, test: t.Data) -> Tuple[t.Data, t.Data]:

        # # Handle non-collection cases.  <- makes no sense
        # if not self.enhance and not self.onmodel:
        #     return train_collection, test_collection
        # train_iterator = train_collection if self.enhance else repeat(None)
        # train_iterator = train_collection if self.enhance else repeat(None)

        # Consume iterators.
        for _ in zip(train.stream, test.stream):
            pass

        if self.hasenhancer:  # TODO: I am not sure these IFs are the right approach...
            train = train.updated((), stream=None)
        if self.hasmodel:  # TODO: ... I've put them here because of streams.
            test = test.updated((), stream=None)
        return train, test

        # As @property is not recognized, mypy raises an error saying that this
        # property coll.data does not exist.
