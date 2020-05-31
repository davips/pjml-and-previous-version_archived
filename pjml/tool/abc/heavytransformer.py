# from abc import ABC
# from typing import Union
#
# from pjdata.collection import Collection
# from pjdata.data import Data
# from pjml.tool.abc.transformer import Transformer
# from pjml.tool.model.model import Model
# from pjml.tool.model.specialmodel import FailedModel, EarlyEndedModel
#
#
# class HeavyTransformer(Transformer, ABC):
#     from pjdata.specialdata import NoData
#
#     def apply(self, data: Union[type, Data] = NoData, exit_on_error=True):
#         if data.isfrozen:
#             return EarlyEndedModel(self, data, data)
#         if data.allfrozen:
#             return EarlyEndedModel(self, data, data.frozen)
#         # TODO: We should check if this 'if' is necessary, because when a model
#         #  fail a FailedModel with frozen data is returned
#         if data.failure:
#             return FailedModel(self, data, data)
#
#         self._check_nodata(data, self)
#
#         # Disable warnings, measure time and make the party happen.
#         self._handle_warnings()  # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         start = self._cpu()
#         try:
#             # Aqui, passa-se _exit_on_error para self de forma que
#             # implementadores de conteineres possam acessar o valor
#             # dentro de
#             # _apply_impl e repassar aos contidos. TODO: Mesmo p/ max_time?
#             self._exit_on_error = exit_on_error
#
#             model = self._limit_by_time(
#                 function=self._apply_impl,
#                 data=data,
#                 max_time=self.max_time
#             )
#
#             # Check result type.
#             if not isinstance(model, Model):
#                 raise Exception(f'{self.name} does not handle {type(model)}!')
#         except Exception as e:
#             self._handle_exception(e, exit_on_error)
#             applied = data.updated(
#                 self.transformations('a'), failure=str(e), frozen=True
#             )
#             self._check_history(data, applied, self.transformations('a'))
#             return FailedModel(self, data, applied)
#             # TODO: é possível que um container não complete o try acima?
#             #  Caso sim, devemos gerar um ContainerModel aqui?
#
#         # TODO: put time_spent inside data (as a "volatile" matrix)?
#         time_spent = self._cpu() - start
#         self._dishandle_warnings()  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#         self._check_history(data, model.data, self.transformations('a'))
#         return model
