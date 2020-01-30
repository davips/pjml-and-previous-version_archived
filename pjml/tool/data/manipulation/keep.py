from pjml.tool.common.configurablecontainer1 import ConfigurableContainer1


# def keep(*args, engine="dump", settings=None, components=None):
#     if components is None:
#         components = args
#     """Shortcut to create a ConfigSpace for Cache."""
#     node = Node(params={'engine': FixedP(engine), 'settings': FixedP(
#     settings)})
#     return SuperCS(Cache.name, Cache.path, components, node)

class Keep(ConfigurableContainer1):
    """Preserve original values of the given fields."""

    # TODO: implement __new__ to generate a CS

    def __init__(self, *args, fields=None, transformers=None):
        if transformers is None:
            transformers = args
        if fields is None:
            fields = ['X', 'Y']
        config = self._to_config(locals())
        del config['args']
        super().__init__(config)

        self.fields = fields
        self.model = fields

    def _apply_impl(self, data):
        return self._step(self.transformer.apply, data)

    def _use_impl(self, data):
        return self._step(self.transformer.use, data)

    def _step(self, f, data):
        matrices = {k: data.fields_safe(k, self) for k in self.fields}
        new_matrices = f(data).matrices
        new_matrices.update(matrices)
        return data.updated(self._transformations(), **new_matrices)
