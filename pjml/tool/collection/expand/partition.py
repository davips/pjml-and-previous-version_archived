from functools import lru_cache
from typing import Optional, List

from pjdata.content.data import Data
from pjml.tool.abc.mixin.component import Component
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.repeat import Repeat


class Partition(Component):
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

    @lru_cache()
    def _enhancer_info(self, data: Data):
        return self.transformer.enhancer.info(data)

    @lru_cache()
    def _model_info(self, data: Data):
        return self.transformer.model(data).info

    def _enhancer_func(self):
        return self.transformer.enhancer.transform

    def _model_func(self, data: Data):
        return self.transformer.model(data).transform

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError
