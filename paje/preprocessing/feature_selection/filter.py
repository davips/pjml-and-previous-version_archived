import abc
from paje.base.component import Component


class Filter(Component):
    """ Filter base class"""

    @abc.abstractmethod
    def rank(self):
        """Todo the docs string
        """

    @abc.abstractmethod
    def score(self):
        """Todo the docs string
        """
