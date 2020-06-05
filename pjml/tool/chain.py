from functools import lru_cache
from itertools import tee
from typing import Optional, Tuple, Dict, List, Callable, Any

from pjdata.aux.util import DataCollTupleT, DataT, flatten
from pjdata.collection import Collection
from pjdata.specialdata import NoData
from pjdata.step.transformation import Transformation
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

    def dual_transform(
        self, prior: DataCollTupleT = NoData, posterior: DataCollTupleT = NoData
    ) -> Tuple[DataCollTupleT, DataCollTupleT]:
        print(self.__class__.__name__, " dual transf (((")
        for trf in self.transformers:
            prior, posterior = trf.dual_transform(prior, posterior)
        return prior, posterior

    def _enhancer_info(self, data: DataT = None) -> Dict[str, Any]:
        return {"enhancers": [trf.enhancer for trf in self.transformers]}

    def _enhancer_func(self) -> Callable[[DataT], DataT]:
        enhancers = self._enhancer_info()["enhancers"]

        def transform(prior):
            for enhancer in enhancers:
                prior = enhancer.transform(prior)
            return prior

        return transform

    def _model_info(self, train: DataT) -> Dict[str, Any]:
        models = []
        for trf in self.transformers:
            if isinstance(train, Collection):
                iter0, iter1 = tee(train.iterator)
                prior0 = Collection(
                    iter0,
                    train.finalizer,
                    debug_info="compo" + self.__class__.__name__ + " pri",
                )
                prior1 = Collection(
                    iter1,
                    train.finalizer,
                    debug_info="compo" + self.__class__.__name__ + " pos",
                )
            else:
                prior0 = prior1 = train
            print("                   gera modelo", trf.name)
            models.append(trf.model(prior0))

            print("      melhora dado pro próximo modelo", trf.name)
            train = trf.enhancer.transform(prior1)
        return {"models": models}

    def _model_func(self, train: DataT) -> Callable[[DataT], DataT]:
        models = self._model_info(train)

        def transform(test: DataT):
            c = 0
            for model in models["models"]:
                print("                 USA modelo", c, model)
                test = model.transform(test)
                c += 1
            return test

        return transform

    @lru_cache()
    def transformations(self, step: str, clean: bool = True) -> List[Transformation]:
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

    def __str__(self, depth=""):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.transformers:
            txts.append(t.__str__(depth))
        return "\n".join(txts)
