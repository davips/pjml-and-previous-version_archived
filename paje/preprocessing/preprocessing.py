from abc import abstractmethod
from paje.base.component import Component


class Preprocessing(Component):
    """ Preprocessing base class"""

    @abstractmethod
    def fit(self, X, y=None):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """Todo the doc string
        """
        pass
