import hashlib
from abc import ABC, abstractmethod

import _pickle as pickle
import blosc
import zstd as zs

import blosc


# @profile
def uuid(packed_content):
    """
    Generates a UUID for any reasonable finite universe.
    It is preferred to generate such MD5 on compressed data,
    since MD5 is much slower for bigger data than the compression itself.
    :param packed_content: packed Data of Xy... or a JSON dump of Component args
    :return: currently a MD5 hash in hex format
    """
    return hashlib.md5(packed_content).hexdigest()


# @profile
def pack_comp(obj):
    return blosc.compress(pickle.dumps(obj),
                          shuffle=blosc.NOSHUFFLE, cname='zstd', clevel=3)


def pack_data(obj):
    return zs.compress(pickle.dumps(obj))


# @profile
def unpack_comp(dump):
    return pickle.loads(blosc.decompress(dump))


def unpack_data(dump):
    return pickle.loads(zs.decompress(dump))


def zip_array(X):
    """
    Attempt to zip faster than with pack, but benchmarks are still needed.
    ps. 1
    Parameters optimized for digits dataset. 115008 rows, 64 attrs
    ps. 2
    The hope of speed gains with this method is probably not worth the
    trouble of applying it to individual parts of Data.
    :param X:
    :return:
    """
    return blosc.compress(X.reshape(1, 115008), cname='blosclz',
                          shuffle=blosc.BITSHUFFLE)


def unzip_array(zipped):
    return blosc.decompress(zipped)



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
