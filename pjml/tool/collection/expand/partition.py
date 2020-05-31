from functools import lru_cache
from typing import Optional, List, Dict, Any, Tuple, Iterator

from pjml.tool.abc.nonfinalizer import NonFinalizer
from pjml.tool.collection.expand.repeat import Repeat
from pjdata.data import Data
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.tool.chain import Chain


class Partition(NonFinalizer, Component):
    """Class to perform, e.g. Expand+kfoldCV.

    This task is already done by function split(),
    but if performance becomes a concern, this less modular solution is a
    good choice.

    TODO: the current implementation is just an alias for the nonoptimized
        previous solution.
    """

    def __init__(
            self,
            split_type: str = 'cv',
            partitions: int = 10,
            test_size: float = 0.3,
            seed: int = 0,
            fields: Optional[List[str]] = None,
            **kwargs
    ):
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(
            locals())  # todo: kwargs is going to locals, em outros comp tb!!!

        # config cleaning.
        if split_type == "cv":
            del config['test_size']
        elif split_type == "loo":
            del config['partitions']
            del config['partition']
            del config['test']
            del config['seed']
        elif split_type == 'holdout':
            pass
        else:
            raise Exception('Wrong split_type: ', split_type)

        super().__init__(config, **kwargs)
        from pjml.macro import tsplit
        self.transformer = Chain(
            Repeat(),
            tsplit(split_type, partitions, test_size, seed, fields)
        )

    @property
    def finite(self):
        return True

    def iterator(self, train: Data, test: Data) -> Iterator[Tuple[Data, Data]]:
        # TODO: not barely optimized.
        return zip(self.enhancer.transform(train),
                   self.model(train).transform(test))

    @lru_cache()
    def enhancer_info(self):
        return self.transformer.enhancer.info

    @lru_cache()
    def model_info(self, data):
        return self.transformer.model(data).info

    def enhancer_func(self):
        return self.transformer.enhancer.transform

    def model_func(self, data):
        return self.transformer.model(data).transform

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError
