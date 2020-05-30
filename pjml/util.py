from typing import NewType, Union, Tuple

from pjdata.collection import Collection
from pjdata.data import Data
from pjdata.specialdata import NoData

TDatasTuple = Union[Tuple[NoData, ...], Tuple[Data, ...],
                    Tuple[Collection, ...],
                    NoData, Data, Collection]

TDatas = Union[NoData, Data, Collection]


def flatten(lst):
    return [item for sublist in lst for item in sublist]
