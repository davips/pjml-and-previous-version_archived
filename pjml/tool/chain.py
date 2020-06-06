from functools import lru_cache
from itertools import tee
from typing import Optional, Tuple, Dict, List, Callable, Any

import pjdata.types as t
from pjdata.aux.util import flatten
from pjdata.content.collection import Collection
from pjdata.content.specialdata import NoData
from pjdata.transformer import Transformer
from pjml.config.description.cs.chaincs import TChainCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component


class Chain(MinimalContainerN):
    """Chain the execution of the given transformers.

    Each arg is a transformer. Optionally, a list of them can be passed as a
    named arg called 'transformers'."""

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

    def dual_transform( # TODO: type overload
            self,
            prior: t.DataOrCollOrTup = NoData,
            posterior: t.DataOrCollOrTup = NoData
    ) -> Tuple[t.DataOrCollOrTup, t.DataOrCollOrTup]:
        print(self.__class__.__name__, ' dual transf (((')
        for trf in self.transformers:
            prior, posterior = trf.dual_transform(prior, posterior)
        return prior, posterior

    @lru_cache()
    def _enhancer_info(self, data: Optional[t.Data]=None) -> Dict[str, Any]:
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
            if isinstance(data, Collection):
                iter0, iter1 = tee(data.iterator)
                prior0 = Collection(
                    iter0, data.finalizer,
                    debug_info='compo' + self.__class__.__name__ + ' pri')
                prior1 = Collection(
                    iter1, data.finalizer,
                    debug_info='compo' + self.__class__.__name__ + ' pos')
            else:
                prior0 = prior1 = data
            print('                   gera modelo', trf.name)
            models.append(trf.model(prior0))

            print('      melhora dado pro próximo modelo', trf.name)
            data = trf.enhancer.transform(prior1)
        return {'models': models}

    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Data]:
        models = self._model_info(data)

        def transform(posterior: t.Data):
            c = 0
            for model in models['models']:
                print('                 USA modelo', c, model)
                posterior = model.transform(posterior)
                c += 1
            return posterior

        return transform

    @lru_cache()
    def transformations(
            self,
            step: str,
            clean: bool = True
    ) -> List[Transformer]:
        lst = []
        for transformer in self.transformers:
            transformations = transformer.transformations(step, clean=False)
            lst.append(transformations)
        result = flatten(lst)
        if clean:
            lst = []
            previous = None
            for transformation in result + [None]:
                if previous and previous.name == "Sink":
                    lst = []
                lst.append(transformation)
                previous = transformation
            result = lst[:-1]
        return result

    def __str__(self, depth=''):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.transformers:
            txts.append(t.__str__(depth))
        return '\n'.join(txts)
