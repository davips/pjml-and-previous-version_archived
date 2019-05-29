import codecs
import hashlib
import time
import traceback
from abc import ABC, abstractmethod

import _pickle as pickle

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.evaluator.time import time_limit


# @profile
def uuid(description):
    # TODO: compact with zlib uglyly to reduce the size by half
    return hashlib.md5(description).hexdigest()  # double time of pickle (moz.)


# @profile
def pack(data):
    dump = pickle.dumps(data)  # irrelevant time (mozilla set)
    return codecs.encode(dump, "base64")  # 5X slower than decode (mozilla set)


# @profile
def unpack(dump):
    decoded = codecs.decode(dump, "base64")
    return pickle.loads(decoded)  # irrelevant time (mozilla set)


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_result(self, component, data):
        pass

    @abstractmethod
    def lock(self, component, test):
        pass

    @abstractmethod
    def get_component_dump(self, component):
        pass

    @abstractmethod
    def get_component(self, component, just_check_exists=False):
        pass

    @abstractmethod
    def result_exists(self, component, train, test):
        pass

    @abstractmethod
    def data_exists(self, data):
        pass

    @abstractmethod
    def get_data(self, data):
        pass

    @abstractmethod
    def store_dset(self, data):
        pass

    @abstractmethod
    def store(self, component, test, testout):
        pass

    # @profile
    def get_or_run(self, component, train, test, f):
        # TODO: Repeated calls to this function with the same parameters can
        #  be memoized, to avoid network delays, for instance.
        testout, time_spent, failed, locked = self.get_result(component,
                                                              train, test)
        print('failed', failed)
        if failed is not None:
            component.failed = failed
            component.locked = locked
            if locked:
                component.warning('Already locked!')
        else:
            # Process data...
            testout, time_spent = f(train, test)

            # Store result.
            print('memoizing results...  failed?:', component.failed)
            self.store(component, train, test, testout, time_spent)
            print('memoized!')
            print()
        return testout, time_spent

    @abstractmethod
    def lock(self, component, train, test):
        pass

