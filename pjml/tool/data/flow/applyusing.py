from pjml.config.cs.containercs import ContainerCS
from pjml.tool.common.nonconfigurablecontainer1 import NonConfigurableContainer1


def applyusing(*args, components=None):
    if components is None:
        components = args
    return ContainerCS(ApplyUsing.name, ApplyUsing.path, components)


class ApplyUsing(NonConfigurableContainer1):
    """Run a 'use' step right after an 'apply' one.

    Useful to calculate training error in classifiers, which would otherwise
    return PhantomData in the 'apply' step."""

    def _apply_impl(self, data):
        self.transformer.apply(data, self._exit_on_error)
        self.model = self.transformer
        return self.transformer.use(data, self._exit_on_error)

    def _use_impl(self, data):
        return self.transformer.use(data, self._exit_on_error)

    def _transformations(self, step=None, training_data=None):
        if training_data is None:
            training_data = self._last_training_data
        return self.transformer._transformations('u', training_data)
