from typing import List

from pjdata import types as t
from pjdata.transformer.transformer import Transformer

from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.tool.abs import component as co
from pjml.tool.abs.component import Component
from pjml.tool.abs.macro import Macro
from pjml.tool.chain import Chain
from pjml.tool.stream.expand.repeat import Repeat


class Partition(Macro):
    """Class to perform, e.g. Expand+kfoldCV.

    This task is already done by function split(),
    but if performance becomes a concern, this less modular solution is a
    good choice.

    TODO: the current implementation is just an alias for the nonoptimized
        previous solution.
    """

    def __init__(self, split_type: str = "cv", partitions: int = 10, test_size: float = 0.3, seed: int = 0,
                 fields: str = "X,Y", **kwargs):
        config = self._to_config(locals())

        # config cleaning.
        if split_type == "cv":
            del config["test_size"]
        elif split_type == "loo":
            del config["partitions"]
            del config["partition"]
            del config["test"]
            del config["seed"]
        elif split_type == "holdout":
            pass
        else:
            raise Exception("Wrong split_type: ", split_type)

        from pjml.macro import split

        self._transformer = Chain(Repeat(), split(split_type, partitions, test_size, seed, fields))
        super().__init__(config, **kwargs)

    @property
    def component(self) -> co.Component:
        return self._transformer

    @classmethod
    def _cs_impl(cls):
        # TODO: Implement this cs
        params = {}
        return CS(nodes=[Node(params=params)])
