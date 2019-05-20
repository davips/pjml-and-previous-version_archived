import codecs
import hashlib
import pickle
from abc import ABC, abstractmethod

from paje.base.exceptions import ExceptionInApplyOrUse


def uuid(description):
    # TODO: compact with zlib uglyly to reduce the size by half
    return hashlib.md5(description).hexdigest()


def pack(data):
    return codecs.encode(pickle.dumps(data), "base64")


def unpack(dump):
    return pickle.loads(codecs.decode(dump, "base64"))


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @abstractmethod
    def get_result(self, component, train, test):
        pass

    @abstractmethod
    def store(self, component, train, test, trainout, testout):
        pass

    def get_or_else(self, component, train, test):
        """
        Results memoization: only output Data is stored for now
        :param component:
        :param train:
        :param test:
        :param f:
        :return:
        """
        # TODO: Repeated calls to this function with the same parameters can
        #  be memoized, to avoid network delays, for instance.
        # TODO: insert time spent
        trainout, testout = self.get_result(component, train, test)
        if trainout is None:
            print('memoizing results...')
            # TODO: is it useful to store the dump of the component?

            # storing only (test and train) predictions: 5kB / row
            # (1 pipeline w/ 3-fold CV = 6 rows)
            try:
                # TODO: failed pipeline should store fake bad predictions
                component.apply(train)
                trainout, testout = component.use(train), component.use(test)
            except Exception as e:
                print('shape train:', train.X.shape, train.y.shape)
                print('shape test:', test.X.shape, test.y.shape)
                raise ExceptionInApplyOrUse(e)

            # Store result.
            self.store(component, train, test, trainout, testout)

        return trainout, testout

    # @abstractmethod
    # def get_model(self, component, train):
    #     """
    #     Extract model from database.
    #     :param component:
    #     :param train:
    #     :return:
    #     """
