# from pjml.config.description.cs.containercs import ContainerCS
#
# from pjml.tool.abc.minimalcontainer import MinimalContainer1
# from pjml.tool.abc.transformer import UTransformer
#
#
# class Wrap(MinimalContainer1):
#     def __new__(cls, *args, transformers=None):
#         """Shortcut to create a ConfigSpace."""
#         if transformers is None:
#             transformers = args
#         if all([isinstance(t, UTransformer) for t in transformers]):
#             return object.__new__(cls)
#         return ContainerCS(Wrap.name, Wrap.path, transformers)
#
#     def _apply_impl(self, data):
#         return self.transformer.apply(data)
#
#     def _use_impl(self, data, **kwargs):
#         pass
#
#     @property
#     def wrapped(self):
#         return self
#
#     def transformations(self, step, clean=True):
#         return self.transformer.transformations(step, clean)
