from abc import ABC, abstractmethod


class AutoML():

    @abstractmethod
    def fit(self, fdata):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def predict(self, data):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def transform(self, data):
        """Todo the doc string
        """
        pass
