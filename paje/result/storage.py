import codecs
import hashlib
import pickle
from abc import ABC, abstractmethod


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @abstractmethod
    def get_or_else(self, component, train, f, test=None):
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


def uuid(description):
    # TODO: compact with zlib uglyly
    return hashlib.md5(description).hexdigest()
