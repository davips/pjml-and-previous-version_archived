from abc import ABC

import numpy

import pjdata.types as t
from pjdata.mixin.serialization import withSerialization
from pjdata.transformer.enhancer import DSStep
from pjdata.transformer.pholder import PHolder
from pjml.tool.abs.mixin.noinfo import withNoInfo
from pjml.tool.data.evaluation.abs.abstractsplit import AbstractSplit


class TsSplit(AbstractSplit, ABC):
    def __init__(self, **kwargs):
        outerself = self

        class Step(withNoInfo, DSStep):  # TODO: info?
            def __init__(self, component: withSerialization, data=None, *args):
                super().__init__(component, *args)
                self._data = data  # Class Component will pass train here, because Step will be understood as Model.

            def _transform_impl(self, data: t.Data) -> t.Result:
                if data != self._data:
                    raise Exception(f"Split needs the same data object at training and test!{data.id}!={self._data.id}")
                zeros = numpy.zeros(data.field(outerself.fields[0], outerself).shape[0])
                partitions = list(outerself.algorithm.split(X=zeros, y=zeros))
                return outerself._split(data, partitions[outerself.partition][1])

        super().__init__(enhancer_cls=PHolder, model_cls=Step, **kwargs)
