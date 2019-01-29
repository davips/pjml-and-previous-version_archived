import abc

class Preprocessing(metaclass=abc.ABCMeta):
    """ Preprocessing base class"""

    @abc.abstractmethod
    def fit(self, X, y):
        """A reference implementation of a fitting function for a preprocessing method."""

    @abc.abstractmethod
    def transform(self, X, y):
        """A reference implementation of a transform for a preprocessing method."""
