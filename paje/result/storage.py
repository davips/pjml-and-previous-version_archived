import codecs
import hashlib
import pickle
import time
import traceback
from abc import ABC, abstractmethod

from sklearn.dummy import DummyClassifier

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
    def get_model(self, component, train, test):
        pass

    @abstractmethod
    def store(self, component, train, test, trainout, testout, time_spent):
        pass

    def get_or_run(self, component, train, test):
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
            # storing only (test and train) predictions: 5kB / row
            # (1 pipeline w/ 3-fold CV = 6 rows) = 30kB / pipe
            # storing complete test and train data and model: 83kB / row
            # = 500kB / pipe

            # storing also args and sets: 1MB / pipe
            # same as above, but storing nothing as model: 720kB / pipe
            start = time.clock()
            try:
                component.apply(train)
                trainout, testout = component.use(train), component.use(test)
            except Exception as e:
                # Fake predictions for curated errors.
                print('Trying to circumvent exception: >' + str(e) + '<')
                msgs = ['All features are either constant or ignored.',  # CB
                        'be between 0 and min(n_samples, n_features)',  # DR*

                        ]
                if any([str(e).__contains__(msg) for msg in msgs]):
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
            self.store(component, train, test, trainout, testout, end - start)

        return trainout, testout
