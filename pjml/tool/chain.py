from itertools import tee
from itertools import tee
from typing import Tuple

import pjdata.types as t
from pjdata.aux.uuid import UUID
from pjdata.content.specialdata import NoData, UUIDData
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.config.description.cs.chaincs import ChainCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.containern import ContainerN


class Chain(ContainerN):
    """Chain the execution of the given components.

    Each arg is a component. Optionally, a list of them can be passed as a
    named arg called 'components'.

    Container with minimum configuration (seed) for more than one component.    """

    def __new__(cls, *args: Component, seed: int = 0, components: Tuple[Component, ...] = None, **kwargs):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(t, Component) for t in components]):
            return object.__new__(cls)
        return ChainCS(*components)

    def __init__(self, *args, seed=0, components=None, enhance=True, model=True):
        if components is None:
            components = args
        outerself = self

        class Enh(Enhancer):

            def _info_impl(self, data):
                return {"enhancers": [c.enhancer for c in outerself.components]}

            def _transform_impl(self, data: t.Data) -> t.Result:
                for enhancer in self.info(data).enhancers:
                    data = enhancer.transform(data)
                return data

        class Mod(Model):

            def _info_impl(self, train):
                models = []
                for comp in outerself.components:
                    if train.stream is None:
                        data0 = data1 = train
                    else:
                        stream0, stream1 = tee(train.stream)
                        # Empty history is acceptable here, because the stream is not changed.
                        data0, data1 = train.updated((), stream=stream0), train.updated((), stream=stream1)
                    models.append(comp.model(data0))
                    train = comp.enhancer.transform(data1)
                return {"models": models}

            def _transform_impl(self, data: t.Data) -> t.Result:
                c = 0
                for model in self.info.models:
                    data = model.transform(data)
                    c += 1
                return data

        super().__init__({}, Enh, Mod, seed, components, enhance, model, deterministic=True)

    def dual_transform(self, train: t.Data = NoData, test: t.Data = NoData) -> Tuple[t.Data, t.Data]:
        for comp in self.components:
            train, test = comp.dual_transform(train, test)
        return train, test

    def _cfuuid_impl(self, data=None):
        """UUID excluding 'model' and 'enhance' flags. Identifies the transformer.

        Chain is a special case, and needs to calculate the uuid based on its internal components.
        """
        # print('chainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
        if data is None:
            uuid = UUID.identity
            for comp in self.components:
                uuid *= comp.enhancer.uuid
                # print('enhhhhhhhhhhhhhh', comp.enhancer.uuid)
            return uuid

        uuid = UUID.identity
        for comp in self.components:
            uuid *= comp.model(data).uuid
            # print('modddddddddddddd', comp.model(data).uuid)
            data = UUIDData(comp.enhancer.uuid * data.uuid)
        return uuid

    # TODO: Chain needs(?) to traverse its subcomponents to build a uuid
    # @lru_cache()
    # def transformations(
    #         self,
    #         step: str,
    #         clean: bool = True
    # ) -> List[Transformer]:
    #     lst = []
    #     for transformer in self.components:
    #         transformations = transformer.transformations(step, clean=False)
    #         lst.append(transformations)
    #     result = flatten(lst)
    #     if clean:
    #         lst = []
    #         previous = None
    #         for transformation in result + [None]:
    #             if previous and previous.name == "Sink":
    #                 lst = []
    #             lst.append(transformation)
    #             previous = transformation
    #         result = lst[:-1]
    #     return result

    def __str__(self, depth=""):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.components:
            txts.append(t.__str__(depth))
        return "\n".join(txts)
