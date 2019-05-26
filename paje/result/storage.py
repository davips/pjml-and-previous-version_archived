import codecs
import hashlib
import _pickle as pickle
import time
import traceback
from abc import ABC, abstractmethod

from sklearn.dummy import DummyClassifier

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
    def setup(self):
        pass

    @abstractmethod
    def get_result(self, component, train, test, just_check_exists,
                   fields_to_store):
        pass

    @abstractmethod
    def get_component_dump(self, component, train, test,
                           just_check_exists=False):
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
    def store(self, component, train, test, trainout, testout,
              time_spent_tr, time_spent_ts, fields_to_store):
        pass

    # @profile
    def get_or_run(self, component, train, test=None,
                   maxtime=60, fields_to_store=None):
        """
        :param component:
        :param train:
        :param test:
        :param maxtime:
        :param fields_to_store:
        :return:
        """
        # TODO: Repeated calls to this function with the same parameters can
        #  be memoized, to avoid network delays, for instance.
        if fields_to_store is None:
            fields_to_store = []
        trainout, testout, failed = \
            self.get_result(component, train, test,
                            fields_to_store=fields_to_store)
        if trainout is not None:
            component.failed = failed
        else:
            # Stats:
            # storing only (test and train) predictions: 5kB / row
            # (1 pipeline w/ 3-fold CV = 6 rows) = 30kB / pipe
            # storing complete test and train data and model: 83kB / row
            # = 500kB / pipe

            # storing also args and sets: 1MB / pipe
            # same as above, but storing nothing as model: 720kB / pipe
            try:
                if component.failed:
                    raise Exception('Pipeline already failed before!')
                with time_limit(maxtime):
                    start = time.clock()
                    component.apply(train)
                    trtime = time.clock() - start
                    trainout = component.use(train)
                    testout = component.use(test)
                    # print(trainout.Xy, 'depois de use(), antes de store')
                    tstime = time.clock() - trtime
            except Exception as e:
                component.failed = True
                trtime = tstime = None
                # Fake predictions for curated errors.
                print('Trying to circumvent exception: >' + str(e) + '<')
                msgs = ['All features are either constant or ignored.',  # CB
                        'be between 0 and min(n_samples, n_features)',  # DR*
                        'excess of max_free_parameters:',  # MLP
                        'Pipeline already failed before!',  # Preemptvely avoid
                        'Timed out!',
                        'Mahalanobis for too big data',
                        'MemoryError',
                        'On entry to DLASCL parameter number',  # Mahala knn
                        'excess of neighbors!',  # KNN
                        ]

                if any([str(e).__contains__(msg) for msg in msgs]):
                    trainout, testout = None, None
                    component.warning(e)
                else:
                    traceback.print_exc()
                    raise ExceptionInApplyOrUse(e)

            # Store result.
            print('memoizing results...')
            self.store(component, train, test, trainout, testout,
                       trtime, tstime, fields_to_store)
            print('memoized!')
            print()
        return trainout, testout
