"""
Shortcuts of common CS/AutoML expressions or ML pipelines.
"""
from typing import List, Optional

from pjml.tool.chain import Chain
from pjml.tool.collection.expand.partition import Partition
from pjml.tool.collection.reduce.summ import RSumm
from pjml.tool.collection.transform.map import Map
from pjml.tool.collection.transform.multi import Multi


def evaluator(*components, function='mean_std', **validation_args):
    return Chain(
        Partition(**validation_args),
        Map(transformers=components),
        RSumm(function=function)
    )


def concat(*transformers):
    # TODO: para que era isso msm?
    pass


def fetch(path):
    """list of files"""
    pass


def switch():
    pass


def tsplit(
        split_type: str = 'cv',
        partitions: int = 10,
        test_size: float = 0.3,
        seed: int = 0,
        fields: Optional[List[str]] = None
) -> Multi:
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
    return Multi(*transformers)
