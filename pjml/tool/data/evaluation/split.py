import numpy
from numpy.random import uniform
from sklearn.model_selection import StratifiedShuffleSplit as HO, \
    StratifiedKFold as SKF, LeaveOneOut as LOO

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import IntP
from pjml.tool.abc.mixin.functioninspector import FunctionInspector
from pjml.tool.abc.transformer import HeavyTransformer
from pjml.tool.model import Model


class Split(HeavyTransformer, FunctionInspector):
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
        return Model(self, data, applied, partitions[self.partition][1])

    def _use_impl(self, data, indices=None, step='u'):
        new_dic = {f: data.field(f, self)[indices] for f in self.fields}
        return data.updated(self.transformations(step), **new_dic)

    @classmethod
    def _cs_impl(cls):
        # TODO complete CS for split; useless?
        params = {
            'partitions': IntP(uniform, low=2, high=10)
        }
        return TransformerCS(Node(params=params))
