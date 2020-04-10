from pjdata.data import Data
from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.lighttransformer import LightTransformer
from pjml.tool.abc.minimalcontainer import MinimalContainer1
from pjml.tool.abc.transformer import Transformer
from pjml.tool.model import Model


class OnlyApply(MinimalContainer1):
    """Does nothing during 'use'."""
    from pjdata.specialdata import NoData

    def __new__(cls, *args, transformers=None):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Transformer) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(OnlyApply.name, OnlyApply.path, transformers)

    def _apply_impl(self, data):
        model = self.transformer.apply(data)
        # return model.updated(self, use_impl=self._use_impl)
        # return Model(self, data, model.data)
        model._use_impl = self._use_impl  # monkeypatch
        model.transformations = self.transformations  # monkeypatch
        return model

    def _use_impl(self, data, **kwargs):
        return data

    def apply(self, data: Data = NoData, exit_on_error=True):
        # We are using here the 'apply()' method from LightTransformer since
        # OnlyApply is less harsh than a real HeavyTransformer.
        return LightTransformer.apply(self, data, exit_on_error=exit_on_error)

    def transformations(self, step, clean=True):
        if step == 'a':
            return self.transformer.transformations(step, clean)
        else:
            return []


class OnlyUse(MinimalContainer1):
    """Does nothing during 'apply'."""
    from pjdata.specialdata import NoData

    def __new__(cls, *args, transformers=None):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, Transformer) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(OnlyUse.name, OnlyUse.path, transformers)

    def _apply_impl(self, data):
        return Model(self, data, data)

    def _use_impl(self, data, **kwargs):
        return self.transformer._use_impl(data, **kwargs)

    def apply(self, data: Data = NoData, exit_on_error=True):
        # We are using here the 'apply()' method from LightTransformer since
        # OnlyUse is less harsh than a real HeavyTransformer.
        return LightTransformer.apply(self, data, exit_on_error=exit_on_error)

    def transformations(self, step, clean=True):
        if step == 'u':
            return self.transformer.transformations(step, clean)
        else:
            return []
