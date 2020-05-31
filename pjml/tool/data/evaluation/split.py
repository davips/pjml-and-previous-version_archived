from functools import lru_cache
from typing import Optional, List, Dict, Any

import numpy
from numpy.random import uniform
from sklearn.model_selection import StratifiedShuffleSplit as HO, \
    StratifiedKFold as SKF, LeaveOneOut as LOO

from pjdata.data import Data
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
from pjml.tool.chain import TChain
from pjml.tool.transformer import TTransformer
from pjml.tool.chain import Chain
from pjml.util import TDatas


class Split(Component, FunctionInspector, NoDataHandler):
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
            split_type: str = 'holdout',
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: Optional[List[str]] = None,
            **kwargs
    ):
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(locals())

        # Using 'self.algorithm' here to avoid 'algorithm' inside config.
        if split_type == "cv":
            self.algorithm = SKF(shuffle=True, n_splits=partitions,
                                 random_state=seed)
            del config['test_size']
        elif split_type == "loo":
            self.algorithm = LOO()
            del config['partitions']
            del config['partition']
            del config['test_size']
            del config['seed']
        elif split_type == 'holdout':
            self.algorithm = HO(n_splits=partitions, test_size=test_size,
                                random_state=seed)
        else:
            raise Exception('Wrong split_type: ', split_type)

        super().__init__(config, **kwargs)

        self.partitions = partitions
        self.partition = partition
        self.test_size = test_size
        self.seed = seed
        self.fields = fields

        self.transformer = Chain(
            SplitTrain(),
            SplitTest()
        )

    def _model_impl(self, prior: TDatas) -> Transformer:
        return self.transformer.model(prior)

    def _enhancer_impl(self) -> Transformer:
        return self.transformer.enhancer   # type: ignore

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError


class SplitTest(Component, FunctionInspector, NoDataHandler):
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
            split_type: str = 'holdout',
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: Optional[List[str]] = None,
            onenhancer: bool = True,
            onmodel: bool = True
    ):
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(locals())

        # Using 'self.algorithm' here to avoid 'algorithm' inside config.
        if split_type == "cv":
            self.algorithm = SKF(shuffle=True, n_splits=partitions,
                                 random_state=seed)
            del config['test_size']
        elif split_type == "loo":
            self.algorithm = LOO()
            del config['partitions']
            del config['partition']
            del config['test_size']
            del config['seed']
        elif split_type == 'holdout':
            self.algorithm = HO(n_splits=partitions, test_size=test_size,
                                random_state=seed)
        else:
            raise Exception('Wrong split_type: ', split_type)

        super().__init__(config)

        self.partitions = partitions
        self.partition = partition
        self.test_size = test_size
        self.seed = seed
        self.fields = fields

    @lru_cache()
    def _info(self, prior: Data) -> Dict[str, Any]:
        zeros = numpy.zeros(prior.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        _posterior = self._split(prior, partitions[self.partition][1],
                                 step='u')
        return {"posterior": _posterior}

    def _enhancer_impl(self):
        return TTransformer(None, None)

    def _model_impl(self, prior: Data) -> Transformer:
        def func(posterior):
            return self._info(posterior)["posterior"]

        return Transformer(func=func, info=None)

    def _split(
            self,
            data: Data,
            indices: List[numpy.ndarray],
            step: str = 'u'
    ) -> Data:
        new_dic = {}
        for f in self.fields:
            try:
                new_dic[f] = data.field(f, self)[indices]
            except Exception as e:
                print(f'\nProblems splitting matrix {f}:', e)
                exit()
        return data.updated(self.transformations(step), **new_dic)

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        # TODO complete CS for split; useless?
        params = {
            'partitions': IntP(uniform, low=2, high=10)
        }
        return TransformerCS(Node(params=params))


class SplitTrain(Component, FunctionInspector, NoDataHandler):
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
            split_type: str = 'holdout',
            partitions: int = 2,
            partition: int = 0,
            test_size: float = 0.3,
            seed: int = 0,
            fields: Optional[List[str]] = None,
            onenhancer: bool = True,
            onmodel: bool = True
    ):
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(locals())

        # Using 'self.algorithm' here to avoid 'algorithm' inside config.
        if split_type == "cv":
            self.algorithm = SKF(shuffle=True, n_splits=partitions,
                                 random_state=seed)
            del config['test_size']
        elif split_type == "loo":
            self.algorithm = LOO()
            del config['partitions']
            del config['partition']
            del config['test_size']
            del config['seed']
        elif split_type == 'holdout':
            self.algorithm = HO(n_splits=partitions, test_size=test_size,
                                random_state=seed)
        else:
            raise Exception('Wrong split_type: ', split_type)

        super().__init__(config)

        self.partitions = partitions
        self.partition = partition
        self.test_size = test_size
        self.seed = seed
        self.fields = fields

    @lru_cache()
    def _info(self, prior: Data) -> Dict[str, Any]:
        zeros = numpy.zeros(prior.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        _prior = self._split(prior, partitions[self.partition][0], step='a')
        return {"prior": _prior}

    def _enhancer_impl(self) -> Transformer:
        def func(prior: Data):
            return self._info(prior)["prior"]

        return Transformer(func=func, info=None)

    def _model_impl(self, data):
        return TTransformer(None, None)

    def _split(
            self,
            data: Data,
            indices: List[numpy.ndarray],
            step: str = 'u'
    ) -> Data:
        new_dic = {}
        for f in self.fields:
            try:
                new_dic[f] = data.field(f, self)[indices]
            except Exception as e:
                print(f'\nProblems splitting matrix {f}:', e)
                exit()
        return data.updated(self.transformations(step), **new_dic)

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        # TODO complete CS for split; useless?
        params = {
            'partitions': IntP(uniform, low=2, high=10)
        }
        return TransformerCS(Node(params=params))
