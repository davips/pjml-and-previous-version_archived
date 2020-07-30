from abc import ABC

import numpy

import pjdata.types as t
from pjdata.transformer.enhancer import DSStep
from pjdata.transformer.pholder import PHolder
from pjml.tool.abs.mixin.noinfo import withNoInfo
from pjml.tool.data.evaluation.abs.abstractsplit import AbstractSplit


class TrSplit(AbstractSplit, ABC):
    def __init__(self, **kwargs):
        outerself = self

        class Step(withNoInfo, DSStep):  # TODO: info?
            def _transform_impl(self, data: t.Data) -> t.Result:
                zeros = numpy.zeros(data.field(outerself.fields[0], outerself).shape[0])
                partitions = list(outerself.algorithm.split(X=zeros, y=zeros))
                return outerself._split(data, partitions[outerself.partition][0])

        super().__init__(enhancer_cls=Step, model_cls=PHolder, **kwargs)
