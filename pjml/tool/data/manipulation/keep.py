#TODO: talvez Keep seja desnecessário,
# já que collection agora mantém referência pra dados originais, entretanto,
# nunca se sabe se outros casos de uso surgirão.
# Poderia existir pra ser atalho de Push/Pop. E esses seriam apelidos de Copy.


# from pjml.config.description.cs.containercs import ContainerCS
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import FixedP
# from pjml.tool.abc.container1 import Container1
#
# # def keep(*args, engine="dump", settings=None, components=None):
# #     if components is None:
# #         components = args
# #     """Shortcut to create a ConfigSpace for Cache."""
# #     node = Node(params={'engine': FixedP(engine), 'settings': FixedP(
# #     settings)})
# #     return SuperCS(Cache.name, Cache.path, components, node)
# from pjml.tool.model.model import Model
# from pjml.tool.abc.transformer import Transformer
#
#
# class Keep(Container1):
#     """Preserve original values of the given fields."""
#
#     # TODO: implement __new__ to generate a CS
#
#     def __new__(cls, *args, fields=None, transformers=None):
#         """Shortcut to create a ConfigSpace."""
#         if transformers is None:
#             transformers = args
#         if all([isinstance(t, Transformer) for t in transformers]):
#             return object.__new__(cls)
#         node = Node(params={'fields': FixedP(fields)})
#         return ContainerCS(Keep.name, Keep.path, transformers, node)
#
#     def __init__(self, *args, fields=None, transformers=None):
#         if transformers is None:
#             transformers = args
#         if fields is None:
#             fields = ['X', 'Y']
#         super().__init__({'fields': fields}, transformers, deterministic=True)
#         self.fields = fields
#
#     def _apply_impl(self, data):
#         # TODO: port it to new schema
#         model = self.transformer.apply(data)
#         applied = self._step(data, model.data, 'a')
#
#         def use_impl(data):
#             inner_used = model.use(data)
#             used = self._step(data, inner_used, 'u')
#             return used
#
#         return Model(applied, self, use_impl)
#
#     def _step(self, data, inner_used, step):
#         matrices = {k: data.field(k, self) for k in self.fields if
#                     k in data.matrices}
#         new_matrices = {} if inner_used is None else inner_used.matrices
#         new_matrices.update(matrices)
#         return data.updated(self.transformations(step), **new_matrices)
