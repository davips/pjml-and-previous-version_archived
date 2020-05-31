import operator
from abc import abstractmethod
from typing import Tuple, Iterator

from itertools import tee

from pjdata.collection import Collection


def unzip_iterator(iterator):
    i1, i2 = tee(iterator)
    return map(operator.itemgetter(0), i1), map(operator.itemgetter(1), i2)


class Batch:
    """Parent mixin for all classes that manipulate collections."""
    onenhancer = onmodel = None  # Come from Component to children classes.

    # TODO: esses métodos parecem gerais o bastante pra ir direto no Component.
    @abstractmethod
    def enhancer_info(self):
        pass

    @abstractmethod
    def model_info(self, data):
        pass

    @abstractmethod
    def enhancer_func(self):
        pass

    @abstractmethod
    def model_func(self, data):
        pass

    # TODO: isso parece geral o bastante pra ir direto no enhancer() no Component.
    def _enhancer_impl(self):
        info = self.enhancer_info()
        transform = self.enhancer_func()
        return TTransformer(
            func=transform,
            info=info
        )

    def _model_impl(self, prior):
        info = self.model_info(prior)
        transform = self.model_func(prior)
        return TTransformer(
            func=transform,
            info=info
        )

    #######################################################################

    @property
    @abstractmethod
    def finite(self):
        pass

    # @abstractmethod
    # def iterators(self, train, test) -> Tuple[Iterator]:
    #     pass

    def dual_transform(self, train, test):
        if not self.onenhancer:
            return train, self.model_func(train)(test)
        if not self.onmodel:
            return self.enhancer_func()(train), test

        def iterator():
            for dtr, dts in zip(train, test):
                yield self.model_func(dtr)(dts), self.enhancer_func()(dtr)

        iterator1, iterator2 = unzip_iterator(iterator())
        coll1 = Collection(iterator1, lambda: train.data, finite=self.finite,
                           debug_info='compo' + self.__class__.__name__ + ' pos')
        coll2 = Collection(iterator2, lambda: test.data, finite=self.finite,
                           debug_info='compo' + self.__class__.__name__ + ' pos')
        return coll1, coll2
