from functools import lru_cache
from itertools import tee
from typing import Optional, Tuple, Dict, Callable, Any

import pjdata.types as t
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.chaincs import ChainCS
from pjml.tool.abs.minimalcontainer import MinimalContainerN
from pjml.tool.abs.component import Component


class Chain(MinimalContainerN):
    """Chain the execution of the given components.

    Each arg is a component. Optionally, a list of them can be passed as a
    named arg called 'components'.
    """

    def __new__(
        cls,
        *args: Component,
        seed: int = 0,
        components: Tuple[Component, ...] = None,
        **kwargs
    ):
        """Shortcut to create a ConfigSpace."""
        if components is None:
            components = args
        if all([isinstance(t, Component) for t in components]):
            return object.__new__(cls)
        return ChainCS(*components)

    def dual_transform(self, train: t.Data = NoData, test: t.Data = NoData) -> Tuple[t.Data, t.Data]:
        for comp in self.components:
            train, test = comp.dual_transform(train, test)
        return train, test

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {'enhancers': [c.enhancer for c in self.components]}

    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        enhancers = self._enhancer_info()['enhancers']

        def transform(prior):
            for enhancer in enhancers:
                prior = enhancer.transform(prior)
            return prior

        return transform

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        models = []
        for trf in self.components:
            if data.stream is None:
                data0 = data1 = data
            else:
                stream0, stream1 = tee(data.stream)
                # Empty history is accepted here, because the stream is not changed.
                data0, data1 = data.updated((), stream=stream0), data.updated((), stream=stream1)
            models.append(trf.model(data0))
            data = trf.enhancer.transform(data1)
        return {'models': models}

    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Data]:
        models = self._model_info(data)

        def transform(posterior: t.Data):
            c = 0
            for model in models['models']:
                # print('                 USA modelo', c, model)
                posterior = model.transform(posterior)
                c += 1
            return posterior

        return transform

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

    def __str__(self, depth=''):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.components:
            txts.append(t.__str__(depth))
        return '\n'.join(txts)
