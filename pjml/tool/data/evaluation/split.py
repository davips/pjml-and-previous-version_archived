from abc import ABC
from functools import lru_cache
from typing import Optional, List, Dict, Any, Callable

import numpy
from numpy.random import uniform
from sklearn.model_selection import (
    StratifiedShuffleSplit as HO,
    StratifiedKFold as SKF,
    LeaveOneOut as LOO,
)

from pjdata.aux.util import Property
from pjdata.content.data import Data
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abc.mixin import component as co
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.defaultenhancer import DefaultEnhancer
from pjml.tool.abc.mixin.defaultmodel import DefaultModel
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.mixin.macro import Macro
from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
from pjml.tool.chain import Chain


class AbstractSplit(Component, FunctionInspector, NoDataHandler, ABC):
    def __init__(
            self,
            split_type: str = "holdout",
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: str = 'X,Y',
            **kwargs
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
        self.fields = fields.split(',')

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
            split_type: str ="holdout",
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: str = 'X,Y',
            **kwargs,
    ):
        config = self._to_config(locals())
        super().__init__(config, **kwargs)

        self._component = Chain(TrSplit(
            split_type=split_type,
            partitions=partitions,
            partition=partition,
            test_size=test_size,
            seed=seed,
            fields=fields,
            **kwargs,
        ), TsSplit(
            split_type=split_type,
            partitions=partitions,
            partition=partition,
            test_size=test_size,
            seed=seed,
            fields=fields,
            **kwargs,
        ))

    @property
    def component(self) -> co.Component:
        return self._component


class TsSplit(DefaultEnhancer, AbstractSplit):
    @lru_cache()
    def _model_info(self, test: Data) -> Dict[str, Any]:
        zeros = numpy.zeros(test.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        test_ = self._split(test, partitions[self.partition][1])
        return {"test": test_, "algorithm": self.algorithm}

    def _model_func(self, prior: Data) -> Callable[[Data], Data]:
        def transform(test: Data) -> Data:
            return self._model_info(test)["test"]

        return transform


class TrSplit(DefaultModel, AbstractSplit):
    def _enhancer_info(self, data: Data) -> Dict[str, Any]:
        zeros = numpy.zeros(data.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        train_ = self._split(data, partitions[self.partition][0])
        return {"train": train_, "algorithm": self.algorithm}

    def _enhancer_func(self) -> Callable[[Data], Data]:
        def transform(train: Data):
            return self._enhancer_info(train)["train"]

        return transform
