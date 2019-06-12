from abc import ABC, abstractmethod


class Cache(ABC):
    """
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    def __init__(self, nested_storage=None):
        """
        This class stores and recovers results from some place.
        :param nested_storage: data will be get first from nested_storage,
        then from current storage. inserts are replicated in both.
        """
        self.nested_storage = nested_storage

    @abstractmethod
    def get_result(self, component, data):
        pass

    @abstractmethod
    def get_component_by_uuid(self, component_uuid, just_check_exists=False):
        pass

    @abstractmethod
    def get_data_by_uuid(self, data, just_check_exists=False):
        pass

    @abstractmethod
    def get_data_by_name(self, name, fields=None, just_check_exists=False):
        pass

    @abstractmethod
    def get_component(self, component, train_data, input_data,
                       just_check_exists=False):
        pass

    @abstractmethod
    def get_finished_names_by_mark(self, mark):
        pass

    @abstractmethod
    def lock(self, component, test, txt=''):
        pass

    @abstractmethod
    def store_data(self, data):
        pass

    @abstractmethod
    def store_result(self, component, test, testout):
        pass
