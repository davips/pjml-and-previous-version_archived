from abc import abstractmethod
from typing import Tuple, Iterator

from pjdata.collection import Collection, AccResult
from pjml.tool.abc.mixin.batch import Batch
from pjml.tool.abc.mixin.component import Component


class Finalizer(Batch, Component):

    def enhancer_func(self):
        def transform(collection):
            def iterator():
                acc = []
                for data in collection:
                    acc.append(self.partial_result(data))
                    yield AccResult(data, acc)

            return Collection(iterator(), self.final_result_func(collection),
                              debug_info='summ')

        return transform

    def model_func(self, data):
        return self.enhancer_func()

    @property
    def finite(self):
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
