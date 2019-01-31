import abc

class Algorithm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def hp_space(self):
        pass
