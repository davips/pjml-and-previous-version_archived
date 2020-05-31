# from cururu.persistence import DuplicateEntryException
# from cururu.storer import Storer
# from pjdata.aux.compression import pack
# from pjdata.data import Data
# from pjml.config.description.cs.transformercs import TransformerCS
# from pjml.config.description.distributions import choice
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import FixedP, CatP
# from pjml.tool.abc.invisible import Invisible
#
#
# class Store(Invisible, Storer):
#     """Store a Data object onto a storage like MySQL, Pickle files, ...
#         as a new dataset. It is a NoOp.
#
#         History and failure are discarded!
#
#         #TODO: componente para guardar resultados de transformação?
#             Já seria o Cache?
#     """
#
#     def __init__(self, name, description='', fields=None,
#                  engine='dump', settings=None):
#         if fields is None:
#             fields = ['X', 'Y']
#         if settings is None:
#             settings = {}
#         config = self._to_config(locals())
#
#         self._set_storage(engine, settings)
#
#         self.model = name, description
#         self.fields = fields
#
#         super().__init__(config, self.model, deterministic=True)
#
#     def _apply_impl(self, data):
#         return self._use_impl(data)
#
#     def _use_impl(self, data, **kwargs):
#         # Enforce a unique name.
#         uuid_ = ''
#         for name in self.fields:
#             uuid_ += uuid(pack(data.field(name, self)))
#         uuid_ = uuid(uuid_.encode())[:7]
#
#         dataset = Dataset(self.model[0] + uuid_, self.model[1])
#         new_data = Data(dataset, **data.matrices)
#         try:
#             self.storage.store(new_data, fields=self.fields,
#                                training_data_uuid=, check_dup=)
#         except DuplicateEntryException as e:
#             print(e)
#         return data
#
#     @classmethod
#     def _cs_impl(cls):
#         params = {
#             'engine': CatP(choice, items=['dump', 'mysql', 'sqlite']),
#             'settings': FixedP({})
#         }
#         return TransformerCS(nodes=[Node(params)])
