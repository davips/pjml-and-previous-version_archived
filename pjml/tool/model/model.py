# from abc import ABC
# from functools import lru_cache
# from typing import Union
#
# from pjdata.data import Data
# from pjdata.mixin.identifyable import Identifyable
# from pjml.tool.abc.mixin.exceptionhandler import ExceptionHandler
# from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
# from pjml.tool.abc.mixin.timers import Timers


# class Model(Identifyable, NoDataHandler, ExceptionHandler, Timers, ABC):
#     """A possibly interpretable ML model able to make predictions, or,
#     more generally, data transformation.
#
#     data_before_apply is a data object
#
#     transformer is needed to define the following model members/values
#     (besides direct calls to transformer):
#         use_impl, uuid, max_time, nodata_handler, transformations,
#         name (based on transformer.longname)
#     """
#     from pjdata.specialdata import NoData
#
#     def __init__(self, transformer, data_before_apply, data_after_apply,
#                  **kwargs):
#         self.transformer = transformer
#         self.data_before_apply = data_before_apply
#         self.data = data_after_apply  # WARN: mutable monkey patched field!
#         self._kwargs = kwargs
#         self.models = None
#
#     # WARN: mutable monkey patched method!
#     def _use_impl(self, data, **kwargs):
#         return self.transformer._use_impl(data, **kwargs)
#
#     @property
#     @lru_cache()
#     def kwargs(self):
#         if self.models is None:
#             return self._kwargs
#         else:
#             _kwargs = {'models': self.models}
#             _kwargs.update(self._kwargs)
#             return _kwargs
#
#     def _uuid_impl(self):
#         # Needless to put mark 'm' to differentiate from uuid of applied data,
#         # since applied data uuid is merged with transformation uuid, not
#         # transformer uuid.
#         return self.data_before_apply.uuid + self.transformer.uuid
#         # TODO: Should Container transformers override uuid in some cases?
#         #  E.g. to avoid storing the same model twice in SGBD?
#         #  Chain(NB()) could have the same uuid as NB()
#         #  Chain(NB(), Report()) could have the same uuid as NB()
#         #  Regarding storage of duplicate Data, it seems not a problem, since
#         #  resulting Data is identified by transformations at the lowest level,
#         #  i.e. without NoOps like Report.
#         #  Should Transformer have a uuid2 based on transformations, we would
#         #  avoid duplicate models for "different" (i.e. containing different
#         #  NoOps) pipelines.
#
#     @property
#     @lru_cache()
#     def name(self):
#         return f'Model[{self.transformer.longname}]'
#
#     def use(self, data: Union[type, Data] = NoData, own_data=False,
#             exit_on_error=True):
#         """Testing step (usually). Predict/transform/do nothing/evaluate/... Data.
#
#         Parameters
#         ----------
#         data
#         own_data
#         exit_on_error
#
#         Returns
#         -------
#         transformed data, normally
#         None, when data is None
#             (probably meaning the pipeline finished before this
#             transformer)
#         same data, but annotated with a failure
#
#         Exception
#         ---------
#         BadComponent
#             Data object resulting history should be consistent with
#             _transformations() implementation.
#         """
#         from pjdata.specialdata import NoData
#
#         data = self.data if own_data else data
#         # Some data checking.
#         if data and data.failure:
#             return data
#         self._check_nodata(data, self.transformer)
#         if data.isfrozen:
#             return data
#         if data.allfrozen:
#             return data.frozen
#
#         # Disable warnings, measure time and make the party happen.
#         self._handle_warnings()  # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         start = self._cpu()
#         try:
#             # Aqui, passa-se _exit_on_error para self de forma que
#             # implementadores de conteineres possam acessar o valor
#             # dentro de
#             # _apply_impl e repassar aos contidos. TODO: Mesmo p/ max_time?
#
#             used = self._limit_by_time(
#                 function=self._use_impl,
#                 data=data,
#                 max_time=self.transformer.max_time,
#                 **self.kwargs
#             )
#
#             # Check result type.
#             isdata_or_collection = isinstance(used, AbstractData)
#             if not isdata_or_collection and used is not NoData:
#                 raise Exception(
#                     f'{self.name} does not handle {type(used)}!\n'
#                     f'Value: {used}'
#                 )
#         except Exception as e:
#             self._handle_exception(e, exit_on_error)
#             used = data.updated(
#                 self.transformations('u'), failure=str(e), frozen=True
#             )
#
#         # TODO: put time_spent inside data (as a "volatile" matrix)?
#         time_spent = self._cpu() - start
#         self._dishandle_warnings()  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#         return self._check_history(data, used, self.transformations('u'))
#
#     def transformations(self, step,
#                         clean=True):  # WARN: mutable monkey patched method!
#         return self.transformer.transformations(step, clean)
#
#     @property
#     @lru_cache()
#     def iscontainer(self):
#         from pjml.tool.model.containermodel import ContainerModel
#         return isinstance(self, ContainerModel)
#
#     def __await__(self):
#         pass
#     # def updated(self, transformer, data_before_apply=None,
#     #             data_after_apply=None, args=None):
#     #     return self._updated(transformer, data_before_apply, data_after_apply,
#     #                          args=args)
#     #
#     # def _updated(self, transformer, data_before_apply=None,
#     #              data_after_apply=None, models=None, args=None):
#     #     from pjml.tool.containermodel import ContainerModel
#     #
#     #     # Update values.
#     #     if transformer is None:
#     #         raise Exception('Transformer cannot be None!')
#     #     if data_before_apply is None:
#     #         data_before_apply = self._uuid_data_before_apply
#     #     if data_after_apply is None:
#     #         data_after_apply = self._data_after_apply
#     #     if args is None:
#     #         args = self._args
#     #
#     #     # Handle ContainerModel specifics.
#     #     if isinstance(self, ContainerModel):
#     #         if models is None:
#     #             models = self.models
#     #         return ContainerModel(
#     #             transformer,
#     #             data_before_apply, data_after_apply,
#     #             models,
#     #             *args
#     #         )
#     #
#     #     return Model(
#     #         transformer,
#     #         data_before_apply, data_after_apply,
#     #         *args
#     #     )
