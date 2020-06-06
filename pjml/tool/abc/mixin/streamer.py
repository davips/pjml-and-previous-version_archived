import operator
from abc import abstractmethod
from typing import Tuple, Iterator

from itertools import tee

from pjdata.content.collection import Collection
from pjdata.content.data import Data


def unzip_iterator(iterator: Iterator) -> Tuple[Iterator, Iterator]:
    i1, i2 = tee(iterator)
    return map(operator.itemgetter(0), i1), map(operator.itemgetter(1), i2)


class Streamer:
    """Parent mixin for all classes that manipulate collections."""
    enhance = model = True  # Come from Component to children classes.
    #TODO: i'm not sure this affects parent class' flags

    # TODO: esses métodos parecem gerais o bastante pra ir direto no Component.
    @abstractmethod
    def _enhancer_info(self, data):
        pass

    @abstractmethod
    def _model_info(self, data):
        pass

    @abstractmethod
    def _enhancer_func(self):
        pass

    @abstractmethod
    def _model_func(self, data):
        pass

    # TODO: isso parece geral o bastante pra ir direto no enhancer() no Component.
    def _enhancer_impl(self) -> Transformer:
        return Transformer(
            func=self.enhancer_func(),
            info=self.enhancer_info
        )

    def _model_impl(self, prior) -> Transformer:
        return Transformer(
            func=self.model_func(prior),
            info=self.model_info(prior)
        )

    #######################################################################

    @Property
    @abstractmethod
    def finite(self):
        pass

    def iterator(self, train: Collection, test: Collection) -> Iterator[Data]:
        for dtr, dts in zip(train, test):
            yield self.enhancer_func()(dtr), self.model_func(dtr)(dts)

    def dual_transform(self, train, test):
        if not self.onenhancer:
            return train, self.model_func(train)(test)
        if not self.onmodel:
            return self.enhancer_func()(train), test

        iterator1, iterator2 = unzip_iterator(self.iterator(train, test))
        coll1 = Collection(iterator1, lambda: train.data, finite=self.finite,
                           debug_info='compo' + self.__class__.__name__ + ' pos')
        coll2 = Collection(iterator2, lambda: test.data, finite=self.finite,
                           debug_info='compo' + self.__class__.__name__ + ' pos')
        return coll1, coll2
