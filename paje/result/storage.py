import codecs
import hashlib
from abc import ABC, abstractmethod

import _pickle as pickle

# @profile
import blosc


def uuid(description):
    # TODO: compact with zlib uglyly to reduce the size by half
    return hashlib.md5(description).hexdigest()  # double time of pickle (moz.)


# @profile
def pack(data):
    dump = pickle.dumps(data)  # irrelevant time (mozilla set)
    # return codecs.encode(dump, "base64")  # 5X slower than decode (mozilla set)
    return dump


# @profile
def unpack(dump):
    # decoded = codecs.decode(dump, "base64")
    # return pickle.loads(decoded)  # irrelevant time (mozilla set)
    return pickle.loads(dump)


def zip_array(X):
    """
    Parameters optimized for digits dataset. 115008 rows, 64 attrs
    :param X:
    :return:
    """
    return blosc.compress(X.reshape(1, 115008), cname='blosclz',
                          shuffle=blosc.BITSHUFFLE)


def unzip_array(zipped):
    return blosc.decompress(zipped)


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def lock(self, component, test):
        pass

    @abstractmethod
    def get_result(self, component, data):
        pass

    @abstractmethod
    def store_data(self, data):
        pass

    @abstractmethod
    def store(self, component, test, testout):
        pass

    @abstractmethod
    def get_data(self, data):
        pass

    @abstractmethod
    def get_data_by_uuid(self, data):
        pass

    # TODO: other useful methods implemented by sql.py,
    #  but not used direcly by Component.
    # @abstractmethod
    # def get_component_dump(self, component):
    #     pass
    #
    # @abstractmethod
    # def get_component(self, component, just_check_exists=False):
    #     pass
    #
    # @abstractmethod
    # def result_exists(self, component, train, test):
    #     pass
    #
    # @abstractmethod
    # def data_exists(self, data):
    #     pass
    #
