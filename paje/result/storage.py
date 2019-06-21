from abc import ABC, abstractmethod


class Cache(ABC):
    """
    The children classes are expected to provide storage in:
     SQLite, remote/local MongoDB or MySQL server.
    """

    def __init__(self, nested_storage=None):
        """
        This class stores and recovers results from some place.
        :param nested_storage: usually the nested storage is local and the
        other is remote. So, all operations occur locally,
        but failed local look ups are tried again remotely.
        If the look up succeeds, then it inserts a replication locally.

        Inserts are replicated in both storages.

        More than one nesting level can exist, good luck doing that.
        """
        self.nested_storage = nested_storage

    def _nested_first(self, f, *args):
        """
        Take a member function name and try first on nested_storage, then on
        self.
        :param f:
        :param kwargs:
        :return:
        """
        if self.nested_storage is not None:
            result = getattr(self.nested_storage, f)(*args)
            if result is not None:
                return result
        return getattr(self, f + '_impl')(*args)

    # TODO: reinsert misses locally
    def get_result(self, component, op, data):
        return self._nested_first('get_result', component, op, data)

    def get_data_by_uuid(self, data_uuid):
        return self._nested_first('get_data_by_uuid', data_uuid)

    def get_data_by_name(self, name, fields=None):
        return self._nested_first('get_data_by_name', name, fields)

    def get_finished_names_by_mark(self, mark):
        return self._nested_first('get_finished_names_by_mark', mark)

    def lock(self, component, op, input_data):
        return self.lock_impl(component, op, input_data)

    def store_data(self, data):
        return self.store_data_impl(data)

    def store_result(self, component, op, input_data, output_data):
        return self.store_result_impl(component, op, input_data, output_data)

    @abstractmethod
    def get_result_impl(self, component, op, data):
        pass

    @abstractmethod
    def get_data_by_uuid_impl(self, data_uuid):
        pass

    @abstractmethod
    def get_data_by_name_impl(self, name, fields=None):
        pass

    @abstractmethod
    def get_finished_names_by_mark_impl(self, mark):
        pass

    @abstractmethod
    def lock_impl(self, component, op, test):
        pass

    @abstractmethod
    def store_data_impl(self, data):
        pass

    @abstractmethod
    def store_result_impl(self, component, op, test, testout):
        pass

    # @abstractmethod
    # def get_component_by_uuid(self, component_uuid):
    #     pass

    # @abstractmethod
    # def get_component(self, component, train_data, input_data):
    #     pass
