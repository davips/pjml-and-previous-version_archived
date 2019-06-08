from abc import ABC, abstractmethod


class Cache(ABC):
    """
    This class stores and recovers results from some place.
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def lock(self, component, test, txt=''):
        pass

    @abstractmethod
    def get_result(self, component, data):
        pass

    @abstractmethod
    def store_data(self, data):
        pass

    @abstractmethod
    def store(self, component, test, testout):
        pass

    @abstractmethod
    def get_data(self, data):
        pass

    @abstractmethod
    def get_data_by_uuid(self, data, just_check_exists=False):
        pass

    @abstractmethod
    def count_results(self, component, data):
        pass

    @abstractmethod
    def get_data_uuid_by_name(self, name, fields='X,y',
                              just_check_exists=False):
        pass

    @abstractmethod
    def get_data_by_name(self, name, fields=None, just_check_exists=False):
        pass

    def get_component_dump(self, component):
        raise NotImplementedError('get model')

    def get_finished(self):
        pass