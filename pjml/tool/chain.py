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
        return result

    def __str__(self, depth=''):
        if not self.pretty_printing:
            return super().__str__()

        txts = []
        for t in self.transformers:
            txts.append(t.__str__(depth))
        return '\n'.join(txts)


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
        for trf in self.transformers:
            prior, posterior = trf.dual_transform(prior, posterior)
        return prior, posterior

    def enhancer(self):
        enhancers = [trf.enhancer() for trf in self.transformers]

        def enhancer_transform(prior):
            for enha in enhancers:
                prior = enha.transform(prior)
            return prior
        return TTransformer(func=enhancer_transform)

    def modeler(self, prior):
        models = [trf.model(prior) for trf in self.transformers]

        def model_transform(posterior):
            for model in models:
                posterior_result = model.transform(posterior)
            return posterior_result
        return TTransformer(func=model_transform)

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

