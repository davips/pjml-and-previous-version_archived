from pjml.tool.abs import component as co
from pjml.tool.abs.macro import Macro
from pjml.tool.chain import Chain
from pjml.tool.data.evaluation.trsplit import TrSplit
from pjml.tool.data.evaluation.tssplit import TsSplit


class Split(Macro):
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
        trsplit = TrSplit(
            split_type=split_type,
            partitions=partitions,
            partition=partition,
            test_size=test_size,
            seed=seed,
            fields=fields,
            **kwargs,
        )
        tssplit = TsSplit(
            split_type=split_type,
            partitions=partitions,
            partition=partition,
            test_size=test_size,
            seed=seed,
            fields=fields,
            **kwargs,
        )
        # HINT: Chain should be in the order below; otherwise, input data will differ for trsplit and tssplit.
        self._component = Chain(tssplit, trsplit)
        super().__init__(config, **kwargs)

    @property
    def component(self) -> co.Component:
        return self._component
