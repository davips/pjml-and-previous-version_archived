import operator
from functools import reduce

from functools import lru_cache
from itertools import dropwhile

from pjdata.specialdata import NoData
from pjml.config.description.cs.chaincs import ChainCS, TChainCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN, TMinimalContainerN
from pjml.tool.abc.mixin.component import TTransformer, TComponent
from pjml.tool.abc.transformer import UTransformer
from pjml.tool.model.containermodel import FailedContainerModel, ContainerModel
from pjml.tool.data.flow.sink import Sink
from pjml.util import flatten


class Chain(MinimalContainerN):
    """Chain the execution of the given transformers.

    Each arg is a transformer. Optionally, a list of them can be passed as a
    named arg called 'transformers'."""

    def __new__(cls, *args, seed=0, transformers=None):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, UTransformer) for t in transformers]):
            return object.__new__(cls)
        return ChainCS(*transformers)

    def _apply_impl(self, data):
        before_data = data
        models = []
        for transformer in self.transformers:
            model = transformer.apply(data, exit_on_error=self._exit_on_error)
            data = model.data
            models.append(model)
            if data and data.failure:
                print(f'Applying subtransformer {transformer} failed! ',
                      data.failure)
                return FailedContainerModel(self, before_data, data, models)

        return ContainerModel(self, before_data, data, models)

    def _use_impl(self, data, models=None):
        for model in models:
            data = model.use(data, exit_on_error=self._exit_on_error)
            if data and data.failure:
                print(f'Using submodel {model} failed! ', data.failure)
                break
        return data

    def transformations(self, step, clean=True):
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
        return tuple(result)

    def __str__(self, depth=''):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.transformers:
            txts.append(t.__str__(depth))
        return '\n'.join(txts)

    def _uuid_impl(self):
        # TODO: override _uuid for other containerNs (Multi)?
        return reduce(operator.mul, [t.uuid for t in self.transformers])


class TChain(TMinimalContainerN):
    """Chain the execution of the given transformers.

    Each arg is a transformer. Optionally, a list of them can be passed as a
    named arg called 'transformers'."""

    def __new__(cls, *args, seed=0, transformers=None):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, TComponent) for t in transformers]):
            return object.__new__(cls)
        return TChainCS(*transformers)

    def dual_transform(self, prior=NoData, posterior=NoData):
        print(self.__class__.__name__, ' dual transf (((')
        for trf in self.transformers:
            prior, posterior = trf.dual_transform(prior, posterior)
        return prior, posterior

    @lru_cache()
    def _info_enhancer(self):
        return {'enhancers': [trf.enhancer for trf in self.transformers]}

    def _enhancer_impl(self):
        enhancers = self._info_enhancer()['enhancers']

        def enhancer_transform(prior):
            for enhancer in enhancers:
                prior = enhancer.transform(prior)
            return prior
        return TTransformer(func=enhancer_transform, info=self._info_enhancer)

    @lru_cache()
    def _info_model(self, prior):
        models = []
        for trf in self.transformers:
            models.append(trf.model(prior))
            prior = trf.enhancer.transform(prior)
        return {'models': models}

    def _model_impl(self, prior):
        models = self._info_model(prior)

        def model_transform(posterior):
            for model in models['models']:
                posterior = model.transform(posterior)
            return posterior
        return TTransformer(func=model_transform, info=models)

    def transformations(self, step, clean=True):
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

