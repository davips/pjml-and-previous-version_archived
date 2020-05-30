from typing import Union, Tuple, Callable, Any, Dict, Set

from pjdata.collection import Collection
from pjdata.data import Data
from pjdata.specialdata import NoData


class Transformer:
    def __init__(
            self,
            func: Callable[[Union[NoData, Data, Collection]],
                           Union[NoData, Data, Collection]],
            info: Union[dict,
                        Callable[[], dict],
                        Callable[[Union[NoData, Data, Collection]], dict]]
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
            data: Union[Tuple[NoData, ...], Tuple[Data, ...],
                        Tuple[Collection, ...], NoData, Data, Collection]
    ) -> Union[Tuple[NoData, ...], Tuple[Data, ...],
               Tuple[Collection, ...], NoData, Data, Collection]:
        if isinstance(data, tuple):
            return tuple((self.safe_func(dt) for dt in data))
        # Todo: We should add exception handling here because self.func can
        #  raise an error
        return self.safe_func(data)

    def safe_func(
            self, data: Union[NoData, Data, Collection]
    ) -> Union[NoData, Data, Collection]:
        if data.isfrozen:
            return data
        return self.func(data)
