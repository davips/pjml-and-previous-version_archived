import abc

class Algorithm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(X, y):

    @abc.abstractmethod
    def hp_space():
