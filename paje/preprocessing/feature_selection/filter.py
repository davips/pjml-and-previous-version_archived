import abc
from paje.base.preprocessing import Preprocessing


class Filter(Preprocessing):
    """ Filter base class"""

    @abc.abstractmethod
    def rank(self):
        """  """


    @abc.abstractmethod
    def score(self):
        """ """
