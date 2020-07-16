from abc import ABC
from typing import Union

from pjdata.content.data import Data
from pjdata.content.specialdata import NoData


class withNoDataHandling(ABC):
    """All components that accept NoData should derive this class after
    deriving Transformer or descendants.

    Should be the last mixin (to avoid overriding attribute 'name')."""

    name = "NoDataHandler"

    def _enforce_nodata(self, data: Union[NoData, Data]) -> None:
        if data is not NoData:
            raise Exception(f"Component {self.name} only accepts NoData. Use Sink before it if needed.")
