from pjdata.finitecollection import FiniteCollection

from pjml.config.description.cs.containercs import ContainerCS
from pjml.tool.abc.minimalcontainer import MinimalContainerN
from pjml.tool.abc.transformer import UTransformer
from pjml.tool.model.containermodel import ContainerModel


class Multi(MinimalContainerN):
    """Process each Data object from a collection with its respective
    transformer."""

    def __new__(cls, *args, transformers=None, seed=0):
        """Shortcut to create a ConfigSpace."""
        if transformers is None:
            transformers = args
        if all([isinstance(t, UTransformer) for t in transformers]):
            return object.__new__(cls)
        return ContainerCS(Multi.name, Multi.path, transformers)

    def _apply_impl(self, collection):
        # TODO: somehow enforce collection for all concurrent components.
        if collection.isfinite and self.size != collection.size:
            raise Exception(
                f'Config space and collection should have the same size '
                f'{self.size} != collection {collection.size}'
            )
        models = []
        datas = []
        for transformer in self.transformers:
            model = transformer.apply(
                next(collection),
                exit_on_error=self._exit_on_error
            )
            datas.append(model.data)
            models.append(model)

        applied = collection.updated(
            self.transformations('a'), datas=datas
        )
        return ContainerModel(self, collection, applied, models)

    def _use_impl(self, collection, models=None):
        if collection.isfinite and self.size != collection.size:
            raise Exception(
                'Config space and collection should have the same '
                f'size {self.size} != collection {collection.size}'
            )
        datas = []
        for model in models:
            data = model.use(
                next(collection),
                exit_on_error=self._exit_on_error
            )
            datas.append(data)
        return collection.updated(self.transformations('u'), datas=datas)
