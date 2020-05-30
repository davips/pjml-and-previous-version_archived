from functools import lru_cache
from typing import Optional, List, Dict, Any

from pjdata.data import Data
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.expand import Expand


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
        config = self._to_config(locals())

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
            Expand(),
            tsplit(split_type, partitions, test_size, seed, fields)
        )

    @lru_cache()
    def _info(self, prior: Data) -> Dict[str, Any]:
        return {'internal_model': self.transformer.model(prior)}

    def _model_impl(self, prior: Data) -> Transformer:
        info = self._info(prior)

        return Transformer(
            func=lambda posterior: info['internal_model'].transform(posterior),
            info=info
        )

    @lru_cache()
    def _info2(self) -> Dict[str, Any]:
        return {'internal_enhancer': self.transformer.enhancer}

    def _enhancer_impl(self) -> Transformer:
        info2 = self._info2()
        return Transformer(
            func=lambda prior: info2['internal_enhancer'].transform(prior),
            info=info2
        )

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError
