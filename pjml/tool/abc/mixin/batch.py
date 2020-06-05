import operator
from abc import abstractmethod, ABC
from itertools import tee
from typing import Tuple, Iterator, Callable

from pjdata.aux.util import Property
from pjdata.collection import Collection
from pjdata.data import Data


def unzip_iterator(iterator: Iterator) -> Tuple[Iterator, Iterator]:
    i1, i2 = tee(iterator)
    return map(operator.itemgetter(0), i1), map(operator.itemgetter(1), i2)


class Batch(ABC):
    """Parent mixin for all classes that manipulate collections."""

    onenhancer = onmodel = True  # Come from Component to children classes.

    @abstractmethod
    def _enhancer_func(self) -> Callable[[Collection], Collection]:
        pass

    @abstractmethod
    def _model_func(self, train_coll: Collection) -> Callable[[Collection], Collection]:
        pass

    @Property
    @abstractmethod
    def finite(self):
        pass

    def iterator(self, train: Collection, test: Collection) -> Iterator[Data]:
        for dtr, dts in zip(train, test):
            yield self._enhancer_func()(dtr), self._model_func(dtr)(dts)

    def dual_transform(self, train, test):
        if not self.onenhancer:
            return train, self._model_func(train)(test)
        if not self.onmodel:
            return self._enhancer_func()(train), test

        iterator1, iterator2 = unzip_iterator(self.iterator(train, test))
        coll1 = Collection(
            iterator1,
            lambda: train.data,
            finite=self.finite,
            debug_info="compo" + self.__class__.__name__ + " pos",
        )
        coll2 = Collection(
            iterator2,
            lambda: test.data,
            finite=self.finite,
            debug_info="compo" + self.__class__.__name__ + " pos",
        )
        return coll1, coll2
