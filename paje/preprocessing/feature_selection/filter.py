import abc
from paje.preprocessing.preprocessing import Preprocessing


class Filter(Preprocessing):
    """ Filter base class"""

    @abc.abstractmethod
    def rank(self):
        """Todo the docs string
        """

    @abc.abstractmethod
    def score(self):
        """Todo the docs string
        """
