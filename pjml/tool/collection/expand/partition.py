from functools import lru_cache

from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.abc.mixin.transformer import TTransformer
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.expand import TExpand


class Partition(TComponent):
    """Class to perform, e.g. Expand+kfoldCV.

    This task is already done by function split(),
    but if performance becomes a concern, this less modular solution is a
    good choice.

    TODO: the current implementation is just an alias for the nonoptimized
        previous solution.
    """

    def __init__(self, split_type='cv', partitions=10, test_size=0.3, seed=0,
                 fields=None, **kwargs):
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(locals())

        # config cleaning.
        if split_type == "cv":
            del config['test_size']
        elif split_type == "loo":
            del config['partitions']
            del config['partition']
            del config['test']
            del config['seed']
        elif split_type == 'holdout':
            pass
        else:
            raise Exception('Wrong split_type: ', split_type)

        super().__init__(config, **kwargs)
        from pjml.macro import tsplit
        self.transformer = Chain(
            TExpand(),
            tsplit(split_type, partitions, test_size, seed, fields)
        )

    @lru_cache()
    def _info(self, prior):
        return {'internal_model': self.transformer.model(prior)}

    def _model_impl(self, prior):
        info = self._info(prior)

        return TTransformer(
            func=lambda posterior: info['internal_model'].transform(posterior),
            info=info
        )

    @lru_cache()
    def _info2(self):
        return {'internal_enhancer': self.transformer.enhancer}

    def _enhancer_impl(self):
        info2 = self._info2()
        return TTransformer(
            func=lambda prior: info2['internal_enhancer'].transform(prior),
            info=info2
        )

    @classmethod
    def _cs_impl(cls):
        raise NotImplementedError

    # TODO: draft of optimized solution:
    # def __init__(self, train_indexes, test_indexes, fields=None):
    #     if fields is None:
    #         fields = ['X', 'Y']
    #     self.config = locals()
    #     self.isdeterministic = True
    #     self.algorithm = fields
    #     self.train_indexes = train_indexes
    #     self.test_indexes = test_indexes
    #
    # def _core(self, data, idxs):
    #     new_dic = {f: data.get_matrix(f)[idxs] for f in self.algorithm}
    #     return data.updated(self._transformation(), **new_dic)
    #
    # def _apply_impl(self, data):
    #     self.model = self.algorithm
    #     return self._core(data, self.train_indexes)
    #
    # def _use_impl(self, data):
    #     return self._core(data, self.test_indexes)
