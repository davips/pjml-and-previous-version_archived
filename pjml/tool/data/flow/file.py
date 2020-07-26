import pjdata.types as t
from functools import lru_cache
from typing import List, Callable, Dict, Any, Tuple

from pjdata.data_creation import read_arff
from pjdata.content.specialdata import NoData
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling


class File(Component, withNoDataHandling):
    """Source of Data object from CSV, ARFF, file.

    TODO: always classification task?
    There will be a single transformation (history) on the generated Data.

    A short hash will be added to the name, to ensure unique names.
    Actually, the first collision is expected after 12M different datasets
    with the same name ( 2**(log(107**7, 2)/2) ).
    Since we already expect unique names like 'iris', and any transformed
    dataset is expected to enter the system through a component,
    12M should be safe enough. Ideally, a single 'iris' be will stored.
    In practice, no more than a dozen are expected.
    """

    def __init__(
            self, name: str, path: str = "./", description: str = "No description.", hashes: str = None, **kwargs,
    ):

        # Some checking.
        if not path.endswith("/"):
            raise Exception("Path should end with /", path)
        if name.endswith("arff"):
            actual_hashes, data = read_arff(path + name, description)
        else:
            raise Exception("Unrecognized file extension:", name)
        if hashes:
            if hashes != actual_hashes:
                raise Exception(f"Provided hashes f{hashes} differs from actual hashes " f"{actual_hashes}!")

        # Unique config for this file.
        config = {
            "name": name,
            "path": path,
            "description": description,
            "hashes": actual_hashes,
        }
        # self._digest = md5digest(serialize(actual_hashes).encode())

        super().__init__(config, deterministic=True, **kwargs)
        self.data = data

    def _enhancer_impl(self) -> Transformer:
        return Enhancer(self, self._transform, lambda _: {})

    def _model_impl(self, train: t.Data) -> Transformer:
        self._enforce_nodata(train)
        return Enhancer(self, self._transform, lambda _: {})

    def _transform(self, data) -> t.Data:
        self._enforce_nodata(data)
        return self.data

    @classmethod
    def _cs_impl(cls) -> CS:
        params = {
            "path": FixedP("./"),
            "name": FixedP("blalbairis.arff"),
            "description": FixedP("No description."),
            "matrices_hash": FixedP("1234567890123456789"),
        }

        # TODO: I think that we should set as follow:
        # TransformerCS(nodes=[Node(params=params)])
        return CS(nodes=[Node(params=params)])
