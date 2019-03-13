from abc import ABC, abstractmethod


class Component(ABC):
    """Todo the docs string
    """
    @abstractmethod
    def hps(self, data):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def apply(self, data):
        """Todo the doc string
        """
        pass

    @staticmethod
    @abstractmethod
    def use(data):
        """Todo the doc string
        """
        pass
