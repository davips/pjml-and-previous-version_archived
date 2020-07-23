# from abc import ABC
#
# from pjdata.step.transformation import Transformation
# from pjml.tool.abc.lighttransformer import LightTransformer
# from pjml.tool.data.algorithm import LightAlgorithm
# from pjml.tool.model.model import Model
#
#
# class Resampler(LightAlgorithm, ABC):
#     """Base class for resampling methods. Not to be confused with Sample."""
#
#     def _apply_impl(self, data):
#         # TODO: generalize this to resample all fields (xyzuvwpq...) or
#         #  create a parameter to define which fields to process
#         sklearn_model = self.algorithm_factory()
#         X, y = sklearn_model.fit_resample(*data.Xy)
#         applied = data.updated(self.transformations('a'), X=X, y=y)
#         return Model(self, data, applied)
#
#     def transformations(self, step, clean=True):
#         if step == 'a':
#             return (Transformation(self, step),)
#         else:
#             return tuple()
