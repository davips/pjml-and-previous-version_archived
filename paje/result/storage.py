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
    def get_setout(self, component, train, setin):
        pass

    @abstractmethod
    def store(self, component, train, setin, setout):
        pass

    def get_or_else(self, component, train, setin, f):
        """
        Results memoization: only output Data is stored for now
        :param component:
        :param train:
        :param setin:
        :param f:
        :return:
        """
        # TODO: Repeated calls to this function with the same parameters can
        #  be memoized, to avoid network delays, for instance.
        # TODO insert time spent
        setout = self.get_setout(component, train, setin)
        if setout is None:
            print('memoizing results...')
            # TODO: is it useful to store the dump of the sets?

            # Apply f()
            try:
                setout = f(setin)
            except Exception as e:
                print('function:', f)
                print('shape train:', train.X.shape, train.y.shape)
                print('shape setin:', setin.X.shape, setin.y.shape)
                raise ExceptionInApplyOrUse(e)

            # Store result.
            # TODO: insert setout
            self.store(component, train, setin, setout)

        return setout

    # @abstractmethod
    # def get_model(self, component, train):
    #     """
    #     Extract model from database.
    #     :param component:
    #     :param train:
    #     :return:
    #     """

    # @abstractmethod
    # def setexists(self, data):
    #     pass
