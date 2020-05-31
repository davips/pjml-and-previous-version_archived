from functools import lru_cache

from itertools import tee
from typing import Optional, Tuple, Dict, List

from pjdata.collection import Collection
from pjdata.specialdata import NoData
from pjdata.step.transformation import Transformation
from pjml.config.description.cs.chaincs import TChainCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.util import flatten, TDatasTuple, TDatas
from pjml.tool.abc.minimalcontainer import TMinimalContainerN
from pjml.tool.abc.mixin.component import TComponent
from pjml.tool.transformer import TTransformer
from pjml.util import flatten



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

    def dual_transform(
            self,
            prior: TDatasTuple = NoData,
            posterior: TDatasTuple = NoData
    ) -> Tuple[TDatasTuple, TDatasTuple]:
        print(self.__class__.__name__, ' dual transf (((')
        for trf in self.transformers:
            prior, posterior = trf.dual_transform(prior, posterior)
        return prior, posterior

    @lru_cache()
    def _info_enhancer(self) -> Dict[str, List[Transformer]]:
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_impl(self) -> Transformer:
        enhancers = self._info_enhancer()['enhancers']

        def enhancer_transform(prior):
            for enhancer in enhancers:
                prior = enhancer.transform(prior)
            return prior

        return Transformer(func=enhancer_transform, info=self._info_enhancer)

    @lru_cache()
    def _info_model(self, prior: TDatas) -> Dict[str, List[Transformer]]:
        models = []
        for trf in self.transformers:
            if isinstance(prior, Collection):
                iter0, iter1 = tee(prior.iterator)
                prior0 = Collection(
                    iter0, prior.finalizer,
                    debug_info='compo'+self.__class__.__name__+' pri')
                prior1 = Collection(
                    iter1, prior.finalizer,
                    debug_info='compo'+self.__class__.__name__+' pos')
            else:
                prior0 = prior1 = prior
            print('                   gera modelo', trf.name)
            models.append(trf.model(prior0))

            print('      melhora dado pro próximo modelo', trf.name)
            prior = trf.enhancer.transform(prior1)
        return {'models': models}

    def _model_impl(self, prior: TDatas) -> Transformer:
        models = self._info_model(prior)

        def model_transform(posterior: TDatas):
            c = 0
            for model in models['models']:
                print('                 USA modelo', c, model)
                posterior = model.transform(posterior)
                c += 1
            return posterior

        return Transformer(func=model_transform, info=models)

    @lru_cache()
    def transformations(
            self,
            step: str,
            clean: bool = True
    ) -> List[Transformation]:
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
