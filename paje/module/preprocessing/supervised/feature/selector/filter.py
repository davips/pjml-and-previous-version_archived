from abc import ABC, abstractmethod

from paje.base.component import Component


class Filter(Component, ABC):
    """ Filter base class"""

    @abstractmethod
    def rank(self):
        """Todo the docs string
        """

    @abstractmethod
    def score(self):
        """Todo the docs string
        """
