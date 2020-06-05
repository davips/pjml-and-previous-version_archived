from abc import abstractmethod
from typing import Tuple, Iterator, Callable

from pjdata.content.collection import Collection, AccResult
from pjml.tool.abc.mixin.streamer import Streamer
from pjml.tool.abc.mixin.component import Component


class Finalizer(Streamer, Component):

    def _enhancer_func(self) -> Callable[[Collection], Collection]:
        def transform(train_coll: Collection) -> Collection:
            def iterator():
                acc = []
                for data in train_coll:
                    acc.append(self.partial_result(data))
                    yield AccResult(data, acc)

            return Collection(iterator(), self.final_result_func(train_coll),
                              debug_info='summ')

        return transform

    def _model_func(self, train_coll: Collection) -> Callable[[Collection], Collection]:
        return self._enhancer_func()

    @property
    def finite(self) -> bool:
        return False

    # def iterators(self, train_collection, test_collection) -> Tuple[Iterator]:
    #     acc = []
    #     for train, test in train_collection, test_collection:
    #         acc.append(self.partial_result(train))
    #         acc.append(self.partial_result(test))
    #         yield AccResult(train, acc), AccResult(test, acc)

    @classmethod
    def _cs_impl(cls):
        pass

    @abstractmethod
    def partial_result(self, data):
        pass

    @abstractmethod
    def final_result_func(self, collection):
        pass
