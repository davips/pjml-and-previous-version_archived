"""
Shortcuts of common CS/AutoML expressions or ML pipelines.
"""
from pjml.tool.chain import Chain
from pjml.tool.collection.expand.partition import TPartition
from pjml.tool.collection.reduce.summ import TRSumm
from pjml.tool.collection.transform.map import Map
from pjml.tool.collection.transform.multi import TMulti


def evaluator(*components, function='mean_std', **validation_args):
    return TChain(
        TPartition(**validation_args),
        TMap(transformers=components),
        TRSumm(function=function)
    )


def concat(*transformers):
    # TODO: para que era isso msm?
    pass


def fetch(path):
    """list of files"""
    pass


def switch():
    pass


def tsplit(split_type='cv', partitions=10, test_size=0.3, seed=0, fields=None):
    """Make a sequence of Data splitters."""
    from pjml.tool.data.evaluation.split import Split
    if fields is None:
        fields = ['X', 'Y']
    transformers = []
    for i in range(partitions):
        s = Split(split_type, partitions, i, test_size, seed, fields)
        transformers.append(s)
    # from pjml.config.description.cs.finitecs import FiniteCS
    # return FiniteCS(trasformers=transformers).sample()
    return TMulti(*transformers)


# def bag(*transformers):
#     """Make a FiniteConfigSpace from a sequence of transformers."""
#     # from pjml.config.description.cs.finitecs import FiniteCS
#     # return FiniteCS(trasformers=transformers)
#     return transformers
