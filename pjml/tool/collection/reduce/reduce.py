from itertools import tee, repeat
from typing import Union, Tuple, Optional

from pjdata.collection import Collection
from pjdata.data import Data
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.util import TDatas, TDatasTuple


class Reduce(Invisible):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        config = {} if config is None else config
        super().__init__(config, deterministic=True, **kwargs)

    def dual_transform(
            self,
            prior_collection: Union[Collection, Tuple[Collection, ...]],
            posterior_collection: Union[Collection, Tuple[Collection, ...]]
    ) -> Union[Tuple[Data, Data], Tuple[Data, Tuple[Data, ...]]]:
        print(self.__class__.__name__, ' dual transf (((')

        # Handle non collection cases.
        if not self.onenhancer:
            prior_collection = repeat(None)
        if not self.onmodel:
            posterior_collection = repeat(None)

        # Consume iterators.
        for _ in zip(prior_collection, posterior_collection):
            pass

        # As @property is not recognized, mypy raises an error saying that this
        # property does not exist.
        return prior_collection.data, posterior_collection.data  # type: ignore

    def _enhancer_impl(self) -> Transformer:
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

        return Transformer(
            func=transform,
            info=None
        )

    def _model_impl(self, prior: TDatasTuple) -> Transformer:
        return self._enhancer_impl()

    @classmethod
    def _cs_impl(cls):
        pass
