from pjdata.data import Data


class TTransformer:
    def __init__(self, func, info):
        self.func = func if func else lambda data: data
        self._info = info

        # Note:
        # Callable returns True, if the object appears to be callable
        # Yes, that appears!
        if callable(self._info):
            self.info = self._info
        elif isinstance(self._info, dict):
            self.info = lambda: self._info
        elif self._info is None:
            self.info = {}
        else:
            raise TypeError('Unexpected info type. You should use callable, '
                            'dict or None.')

    def transform(self, data: Data):  # resolver error
        print('!!!!!!!!!!!!!!!', data, type(self).__name__, type(data))
        if isinstance(data, tuple):
            return tuple((dt.transformedby(self.func) for dt in data))
        # Todo: We should add exception handling here because self.func can
        #  raise an error
        return data.transformedby(self.func)
