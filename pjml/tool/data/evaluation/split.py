from functools import lru_cache

import numpy
from numpy.random import uniform
from sklearn.model_selection import StratifiedShuffleSplit as HO, \
    StratifiedKFold as SKF, LeaveOneOut as LOO

from pjdata.specialdata import NoData
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abc.mixin.component import TComponent, TTransformer
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
from pjml.tool.abc.transformer import DTransformer
from pjml.tool.model.model import Model


class Split(DTransformer, FunctionInspector):
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

    def __init__(self, split_type='holdout', partitions=2, partition=0,
                 test_size=0.3, seed=0, fields=None):
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

    def _apply_impl(self, data):
        zeros = numpy.zeros(data.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        applied = self._use_impl(data, partitions[self.partition][0], step='a')
        return Model(self, data, applied, indices=partitions[self.partition][1])

    def _use_impl(self, data, indices=None, step='u'):
        new_dic = {}
        for f in self.fields:
            try:
                new_dic[f] = data.field(f, self)[indices]
            except Exception as e:
                print(f'\nProblems splitting matrix {f}:', e)
                exit()
        return data.updated(self.transformations(step), **new_dic)

    @classmethod
    def _cs_impl(cls):
        # TODO complete CS for split; useless?
        params = {
            'partitions': IntP(uniform, low=2, high=10)
        }
        return TransformerCS(Node(params=params))


class TSplit(TComponent, FunctionInspector, NoDataHandler):
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

    def __init__(self, split_type='holdout', partitions=2, partition=0,
                 test_size=0.3, seed=0, fields=None):
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
    def _info(self, prior):
        zeros = numpy.zeros(prior.field(self.fields[0], self).shape[0])
        partitions = list(self.algorithm.split(X=zeros, y=zeros))
        _prior = self._split(prior, partitions[self.partition][0], step='a')
        _posterior = self._split(prior, partitions[self.partition][1], step='u')
        return {"prior": _prior, "posterior": _posterior}

    def _model_impl(self, prior):
        def func(posterior=prior):
            if posterior != prior:
                raise Exception('Split expects the same Data (posterior should'
                                'be equal prior).')
            return self._info(prior)["posterior"]
        return TTransformer(func=func, info=self._info(prior))

    def _enhancer_impl(self):
        def func(prior):
            return self._info(prior)["prior"]
        return TTransformer(func=func, info=self._info)

    def _split(self, data, indices=None, step='u'):
        new_dic = {}
        for f in self.fields:
            try:
                new_dic[f] = data.field(f, self)[indices]
            except Exception as e:
                print(f'\nProblems splitting matrix {f}:', e)
                exit()
        return data.updated(self.transformations(step), **new_dic)

    @classmethod
    def _cs_impl(cls):
        # TODO complete CS for split; useless?
        params = {
            'partitions': IntP(uniform, low=2, high=10)
        }
        return TransformerCS(Node(params=params))
