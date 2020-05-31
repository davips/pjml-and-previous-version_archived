# import inspect
#
# from pjml.config.description.cs.containercs import ContainerCS
# from pjml.tool.abc.minimalcontainer import MinimalContainer1
# from pjml.tool.abc.transformer import UTransformer
# from pjml.tool.model.model import Model
#
#
# class ApplyUsing(MinimalContainer1):
#     """Run a 'use' step right after an 'apply' one.
#
#     Useful to calculate training error in classifiers, which would otherwise
#     return PhantomData in the 'apply' step."""
#
#     def __new__(cls, *args, transformers=None):
#         """Shortcut to create a ConfigSpace."""
#         if transformers is None:
#             transformers = args
#         if all([isinstance(t, UTransformer) for t in transformers]):
#             return object.__new__(cls)
#         return ContainerCS(ApplyUsing.name, ApplyUsing.path, transformers)
#
#     def _apply_impl(self, data):
#         model = self.transformer.apply(data, exit_on_error=self._exit_on_error)
#         applied = model.use(data, exit_on_error=self._exit_on_error)
#         # m = Model(self.transformer, data, applied, *model.args)
#         # m = model.updated(self.transformer, data_after_apply=applied)
#         # m = model.updated(self, data_after_apply=applied)
#         model.data = applied  # monkeypatch
#         return model
#
#     def _use_impl(self, data, **kwargs):
#         raise Exception('This should never happen!')
#
#     def transformations(self, step, clean=True):
#         return self.transformer.transformations('u', clean)
