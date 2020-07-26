from abc import ABC
from functools import lru_cache
from typing import List, Dict, Any, Callable

import numpy
from numpy.random import uniform
from sklearn.model_selection import (
    StratifiedShuffleSplit as HO,
    StratifiedKFold as SKF,
    LeaveOneOut as LOO,
)

import pjdata.types as t
from pjdata.content.data import Data
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.pholder import PHolder
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abs import component as co
from pjml.tool.abs.component import Component
from pjml.tool.abs.macro import Macro
from pjml.tool.abs.mixin.functioninspection import withFunctionInspection
from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling
from pjml.tool.chain import Chain


class AbstractSplit(Component, withFunctionInspection, withNoDataHandling, ABC):
    def __init__(
            self,
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

        super().__init__(config, **kwargs)

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


class Split(Macro, Component):
    """Split a given Data field into training/apply set and testing/use set.

    Developer: new metrics can be added just following the pattern '_fun_xxxxx'
    where xxxxx is the name of the new metric.

    Parameters
    ----------
    train_indexes
        Indexes of rows to get from data objects during apply().
    test_indexes
        Indexes of rows to get from data objects during use().
    fields
        Name of the matrices to be modified.
    """

    def __init__(
            self,
            split_type: str = "holdout",
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: str = "X,Y",
            **kwargs,
    ):
        config = self._to_config(locals())
        super().__init__(config, **kwargs)

        self._component = Chain(
            TrSplit(
                split_type=split_type,
                partitions=partitions,
                partition=partition,
                test_size=test_size,
                seed=seed,
                fields=fields,
                **kwargs,
            ),
            TsSplit(
                split_type=split_type,
                partitions=partitions,
                partition=partition,
                test_size=test_size,
                seed=seed,
                fields=fields,
                **kwargs,
            ),
        )

    @property
    def component(self) -> co.Component:
        return self._component


class TrSplit(AbstractSplit, ABC):
    def _enhancer_impl(self) -> Transformer:
        def transform(data: Data):
            zeros = numpy.zeros(data.field(self.fields[0], self).shape[0])
            partitions = list(self.algorithm.split(X=zeros, y=zeros))
            return self._split(data, partitions[self.partition][0])

        return Enhancer(self, transform, lambda _: {"train": "TODO:put doubtful infohere", "algorithm": self.algorithm})

    def _model_impl(self, data: t.Data) -> Transformer:
        return PHolder(self)


class TsSplit(AbstractSplit, ABC):
    def _enhancer_impl(self) -> Transformer:
        return PHolder(self)

    def _model_impl(self, data: t.Data) -> Transformer:
        zeros = numpy.zeros(data.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        testsplit = self._split(data, partitions[self.partition][1])

        def transform(test):
            if data != test:
                raise Exception("Split needs the same data object at training and test!")
            return testsplit

        return Enhancer(self, transform, lambda _: {"test": testsplit, "algorithm": self.algorithm})
