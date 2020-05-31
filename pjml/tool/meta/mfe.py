# import numpy as np
# import pymfe.mfe
#
# from pjml.config.description.cs.emptycs import EmptyCS
# from pjml.tool.abc.lighttransformer import LightTransformer
# from pjml.tool.model.model import Model
#
#
# class MFE(LightTransformer):
#     """Uniformly distribute examples along each attribute to make them
#     independent of scale and unit measure.
#
#     Each attribute value is replaced by the order in which the example is
#     ranked according with that attribute.
#     Applying a normalization after this transformer is recommended."""
#
#     def __init__(self):
#         super().__init__({}, deterministic=True)
#
#     @classmethod
#     def _cs_impl(cls):
#         return EmptyCS()
#
#     def _apply_impl(self, data):
#         applied = self._use_impl(data, step='a')
#         return Model(self, data, applied)
#
#     def _use_impl(self, data, step='u'):
#         pymfe.mfe.MFE().fit(*data.Xy)
#         ft = pymfe.mfe.MFE().extract()
#         return data.updated(self.transformations(step),
#                             M=np.array([ft[1]]), Md=ft[0])
