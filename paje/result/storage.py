import codecs
import hashlib
import pickle
import time
import traceback
from abc import ABC, abstractmethod

from sklearn.dummy import DummyClassifier

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.evaluator.time import time_limit


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
              time_spent_tr, time_spent_ts):
        pass

    def get_or_run(self, component, train, test, maxtime=60):
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
        trainout, testout, failed = \
            self.get_result(component, train, test)
        if trainout is not None:
            component.failed = failed
        else:
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
                    trainout, testout = component.use(train), \
                                        component.use(test)
                    tstime = time.clock() - trtime
            except Exception as e:
                component.failed = True
                # Fake predictions for curated errors.
                print('Trying to circumvent exception: >' + str(e) + '<')
                msgs = ['All features are either constant or ignored.',  # CB
                        'be between 0 and min(n_samples, n_features)',  # DR*
                        'excess of max_free_parameters:',  # MLP
                        'Pipeline already failed before!',  # Preemptvely avoid
                        'Timed out!',
                        ]
                if str(e).__contains__('Pipeline already failed before!'):
                    trtime = tstime = 99999999

                if any([str(e).__contains__(msg) for msg in msgs]):
                    trtime = tstime = 0
                    # We suppose here that all pipelines are for classification.
                    model = DummyClassifier(strategy='uniform')
                    model.fit(*train.xy)
                    zr = model.predict(train.y)
                    zs = model.predict(test.y)
                    trainout, testout = train.updated(z=zr), test.updated(z=zs)
                    component.warning(e)
                else:
                    traceback.print_exc()
                    raise ExceptionInApplyOrUse(e)

            # Store result.
            end = time.clock()
            print('memoizing results...')
            self.store(component, train, test, trainout, testout,
                       trtime, tstime)
            print('memoized!')
            print()
        return trainout, testout
