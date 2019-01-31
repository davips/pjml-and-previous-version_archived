import abc

class Classifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """A reference implementation of a prediction for a classifier."""
        pass
