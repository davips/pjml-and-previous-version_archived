import codecs
import hashlib
import pickle
from abc import ABC, abstractmethod


def uuid(description):
    # TODO: compact with zlib uglyly to reduce the size by half
    return hashlib.md5(description).hexdigest()


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @classmethod
    def pack(cls, data):
        return codecs.encode(pickle.dumps(data), "base64")

    @classmethod
    def unpack(cls, dump):
        return pickle.loads(codecs.decode(dump, "base64"))

    @abstractmethod
    def get_model(self, component, train):
        """
        Extract model from database.
        :param component:
        :param train:
        :return:
        """

    @abstractmethod
    def get_or_else(self, component, train, test, f):
        """
        faster way of memoization: only output Data is stored for now
        :param component:
        :param train:
        :param test:
        :param f:
        :return:
        """
