from typing import Union, Callable, Optional

from pjml.util import TDatas, TDatasTuple


class Transformer:
    def __init__(
            self,
            func: Optional[Callable[[TDatas], TDatas]],
            info: Optional[
                Union[dict,
                      Callable[[], dict],
                      Callable[[TDatas], dict]]]
    ):
        self.func = func if func else lambda data: data

        # Note:
        # Callable returns True, if the object appears to be callable
        # Yes, that appears!
        if callable(info):
            self.info = info
        elif isinstance(info, dict):
            self.info = lambda: info
        elif info is None:
            self.info = lambda: {}
        else:
            raise TypeError('Unexpected info type. You should use, callable, '
                            'dict or None.')

    def transform(
            self,
            data: TDatasTuple
    ) -> TDatasTuple:
        if isinstance(data, tuple):
            return tuple((self.safe_func(dt) for dt in data))
        # Todo: We should add exception handling here because self.func can
        #  raise an error
        return self.safe_func(data)

    def safe_func(
            self, data: TDatas
    ) -> TDatas:
        if data.isfrozen:
            return data
        return self.func(data)
