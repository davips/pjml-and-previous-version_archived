# from pjml.tool.model.model import Model
#
#
# class ContainerModel(Model):
#     def __init__(self, transformer, data_before_apply, data_after_apply, models,
#                  **kwargs):
#         super().__init__(transformer, data_before_apply, data_after_apply,
#                          **kwargs)
#
#         # ChainModel(ChainModel(a,b,c)) should be equal to ChainModel(a,b,c)
#         if len(models) == 1 and models[0].iscontainer:
#             models = models[0].models
#
#         self.models = models
#
#     # def updated(self, transformer, data_before_apply=None,
#     #             data_after_apply=None, models=None, args=None):
#     #     return self._updated(transformer, data_before_apply, data_after_apply,
#     #                          models=models, args=args)
#
#
# class FailedContainerModel(ContainerModel):
#     def __init__(self, transformer, data_before_apply, data_after_apply, models,
#                  **kwargs):
#         super().__init__(transformer, data_before_apply, data_after_apply,
#                          models, **kwargs)
#
#     def _use_impl(self, data, **kwargs):
#         raise Exception(
#             f"A {self.name} model from failed pipelines during apply is not "
#             f"usable!"
#         )
