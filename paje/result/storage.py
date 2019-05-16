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
        # buf = io.BytesIO()
        # joblib.dump(data, buf, compress=('bz2', 9))
        # return memoryview(bytes(buf))

    @classmethod
    def unpack(cls, dump):
        return pickle.loads(codecs.decode(dump, "base64"))
        # buf = io.BytesIO()
        # buf.write(dump)
        # return joblib.load(buf)

    @abstractmethod
    def get_set(self, data_hash):
        """
        Extract data from database.
        :param data_hash:
        :param data:
        :return:
        """

    @abstractmethod
    def setexists(self, data_hash):
        """
        Check if data already exists in database.
        :param data:
        :return:
        """

    @abstractmethod
    def argsexist(self, component):
        """
        Check if component args already exist in database.
        :param component:
        :return:
        """

    @abstractmethod
    def get_model(self, component, train):
        """
        Extract model from database.
        :param component:
        :param train:
        :return:
        """

    @abstractmethod
    def get_or_else(self, component, train, f):
        """
        Retrieve a component instance (usually a pipeline) from database if it exists.
        Otherwise, create it calling f and store it in database.
        :param component:
        :param train:
        :param f: function to execute if component has never been applied before.
        :param test:
        :return:
        """
        pass

    @abstractmethod
    def get_results_or_else(self, component, train, test, f):
        """
        faster way of memoization
        :param component:
        :param train:
        :param test:
        :param f:
        :return:
        """
