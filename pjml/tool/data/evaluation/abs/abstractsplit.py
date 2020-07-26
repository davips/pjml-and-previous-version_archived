from abc import ABC
from typing import List

import numpy
from numpy.random import uniform
from sklearn.model_selection import (
    StratifiedShuffleSplit as HO,
    StratifiedKFold as SKF,
    LeaveOneOut as LOO,
)

from pjdata.content.data import Data
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.functioninspection import withFunctionInspection
from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling


class AbstractSplit(Component, withFunctionInspection, withNoDataHandling, ABC):
    def __init__(
            self,
            enhancer_cls,
            model_cls,
            split_type: str = "holdout",
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: str = "X,Y",
            **kwargs,
    ):
        config = self._to_config(locals())

        # Using 'self.algorithm' here to avoid 'algorithm' inside config.
        if split_type == "cv":
            self.algorithm = SKF(shuffle=True, n_splits=partitions, random_state=seed)
            del config["test_size"]
        elif split_type == "loo":
            self.algorithm = LOO()
            del config["partitions"]
            del config["partition"]
            del config["test_size"]
            del config["seed"]
        elif split_type == "holdout":
            self.algorithm = HO(n_splits=partitions, test_size=test_size, random_state=seed)
        else:
            raise Exception("Wrong split_type: ", split_type)

        super().__init__(config, enhancer_cls=enhancer_cls, model_cls=model_cls, **kwargs)

        self.partitions = partitions
        self.partition = partition
        self.test_size = test_size
        self.seed = seed
        self.fields = fields.split(",")

    def _split(self, data: Data, indices: List[numpy.ndarray]) -> Data:
        new_dic = {}
        for f in self.fields:
            try:
                new_dic[f] = data.field(f, self)[indices]
            except Exception as e:
                print(f"\nProblems splitting matrix {f}:", e)
                exit()
        return new_dic

    @classmethod
    def _cs_impl(cls) -> CS:
        # TODO complete CS for split; useless?
        params = {"partitions": IntP(uniform, low=2, high=10)}
        return CS(Node(params=params))
