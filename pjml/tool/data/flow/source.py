# from cururu.storer import Storer
# from pjml.config.description.cs.transformercs import TransformerCS
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import FixedP
# from pjml.tool.abc.lighttransformer import LightTransformer
# from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
# from pjml.tool.model.model import Model
#
#
# class Source(LightTransformer, NoDataHandler, Storer):
#     """Source of Data object from a storage like MySQL, Pickle files, ... """
#
#     def __init__(self, name, fields=None, engine='dump', settings=None):
#         if fields is None:
#             fields = ['X', 'Y']
#         if settings is None:
#             settings = {}
#         config = self._to_config(locals())
#
#         self._set_storage(engine, settings)
#         lst = self.storage.list_by_name(name)
#         if len(lst) == 0:
#             raise Exception(f'{name} dataset not found!')
#         if len(lst) > 1:
#             raise Exception(f'More than one dataset named {name} found!')
#         if lst[0].name != name:
#             raise Exception(f'Fetched name {lst[0].name} differs '
#                             f'from provided name {name}!')
#         phantom_data = lst[0]
#         data = self.storage.fetch(phantom_data, fields=fields)
#         if data is None:
#             raise Exception('Dataset not found: ', phantom_data.name)
#         self.data = data
#         super().__init__(config, self.data, deterministic=True)
#
#     def _apply_impl(self, data):
#         from pjdata.specialdata import NoData
#         if data is not NoData:
#             raise Exception('Source component needs to be applied with NoData. '
#                             'Use Sink before it if needed.')
#         return Model(self, NoData, self.data)
#
#     def _use_impl(self, data, **kwargs):
#         from pjdata.specialdata import NoData
#         if data is not NoData:
#             raise Exception('Source component needs to be used with NoData. '
#                             'Use Sink before it if needed.')
#         return self.data
#
#     def transformations(self, step, clean=True):
#         return self.data.history.transformations
#
#     @classmethod
#     def _cs_impl(cls):
#         params = {
#             'name': FixedP('iris_OFÆdñO')
#         }
#         return TransformerCS(nodes=[Node(params=params)])
