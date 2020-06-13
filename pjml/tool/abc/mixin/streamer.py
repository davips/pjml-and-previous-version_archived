import operator
from abc import abstractmethod, ABC
from itertools import tee
from typing import Tuple, Iterator, Callable

from pjdata.aux.util import Property
from pjdata.content.collection import Collection
from pjdata.content.data import Data
import pjdata.types as t


def unzip_iterator(iterator: Iterator) -> Tuple[Iterator, Iterator]:
    i1, i2 = tee(iterator)
    return map(operator.itemgetter(0), i1), map(operator.itemgetter(1), i2)


class Streamer(ABC):
    """Parent mixin for all classes that manipulate collections."""
    _enhance = _model = True  # Come from Component to children classes.
    #TODO: i'm not sure this affects parent class' flags

    @abstractmethod
    def _enhancer_func(self) -> Callable[[t.DataOrColl], t.DataOrColl]:
        pass

    @abstractmethod
    def _model_func(self, train: t.DataOrColl) -> Callable[[t.DataOrColl], t.DataOrColl]:
        pass

    @Property
    @abstractmethod
    def finite(self):
        pass

    def iterator(self, train: Collection, test: Collection) -> Iterator[Data]:
        for dtr, dts in zip(train, test):
            yield self._enhancer_func()(dtr), self._model_func(dtr)(dts)

    def dual_transform(self, train, test):
        if not self._enhance:
            return train, self._model_func(train)(test)
        if not self._model:
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
