from functools import lru_cache
from itertools import tee
from typing import Optional, Tuple, Dict, List, Callable, Any

import pjdata.types as t
from pjdata.aux.util import flatten
from pjdata.content.specialdata import NoData
from pjdata.transformer import Transformer
from pjml.config.description.cs.chaincs import TChainCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component


class Chain(MinimalContainerN):
    """Chain the execution of the given transformers.

    Each arg is a transformer. Optionally, a list of them can be passed as a
    named arg called 'transformers'.
    """

    def __new__(
        cls,
        *args: Component,
        seed: int = 0,
        transformers: Optional[Tuple[Component, ...]] = None,
        **kwargs
    ):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Component) for t in transformers]):
            return object.__new__(cls)
        return TChainCS(*transformers)

    def dual_transform(self, train: t.Data=NoData, test: t.Data=NoData) -> Tuple[t.Data, t.Data]:
        for trf in self.transformers:
            train, test = trf.dual_transform(train, test)
        return train, test

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        enhancers = self._enhancer_info()['enhancers']

        def transform(prior):
            for enhancer in enhancers:
                prior = enhancer.transform(prior)
            return prior

        return transform

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        models = []
        for trf in self.transformers:
            if data.stream is None:
                data0 = data1 = data
            else:
                stream0, stream1 = tee(data.stream)
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

    # TODO: Chain needs to report to Cache about its "monster" status
    #  That's because monsters generate two histories:
    #  the predictable short (itself) and the possibly data-dependent long (all that goes inside Data as "bytecodes")
    #  This code commented out below can be useful for that.
    # @lru_cache()
    # def transformations(
    #         self,
    #         step: str,
    #         clean: bool = True
    # ) -> List[Transformer]:
    #     lst = []
    #     for transformer in self.transformers:
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
        for t in self.transformers:
            txts.append(t.__str__(depth))
        return '\n'.join(txts)
