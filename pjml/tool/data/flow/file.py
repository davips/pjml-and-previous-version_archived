from pjdata.aux.serialization import serialize
from pjdata.aux.uuid import UUID
from pjdata.creation import read_arff
from pjdata.transformer.enhancer import DSStep
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling
from pjml.tool.abs.mixin.noinfo import withNoInfo


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

    def __init__(self, name: str, path: str = "./", hashes: str = None, **kwargs):
        # Some checking.
        if not path.endswith("/"):
            raise Exception("Path should end with /", path)
        if name.endswith("arff"):
            actual_hashes, data, _, _ = read_arff(path + name)
        else:
            raise Exception("Unrecognized file extension:", name)
        if hashes:
            if hashes != actual_hashes:
                raise Exception(f"Provided hashes f{hashes} differs from actual hashes " f"{actual_hashes}!")

        # Unique config for this file.
        config = {
            "name": name,
            "path": path,
            "hashes": actual_hashes,
        }

        # self._digest = md5digest(serialize(actual_hashes).encode())

        class Step(withNoInfo, withNoDataHandling, DSStep):
            def _transform_impl(self, nodata):
                self._enforce_nodata(nodata)
                return data

        super().__init__(config, enhancer_cls=Step, model_cls=Step, deterministic=True, **kwargs)

    @classmethod
    def _cs_impl(cls) -> CS:
        params = {
            "path": FixedP("./"),
            "name": FixedP("blalbairis.arff"),
            "matrices_hash": FixedP("1234567890123456789"),
        }

        return CS(nodes=[Node(params=params)])

    def _cfuuid_impl(self, data=None):
        return UUID(serialize(self.config["hashes"]).encode())
