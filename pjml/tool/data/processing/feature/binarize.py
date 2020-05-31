# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
#
# from pjdata.data_creation import nominal_idxs
# from pjml.tool.abc.configless import ConfigLess
# from pjml.tool.model.model import Model


# class Binarize(ConfigLess):
#     def _apply_impl(self, data):
#         applied = self._use_impl(data)
#         return Model(self, data, applied)
#
#     def _use_impl(self, data, **kwargs):
#         # TODO: check Data object compatibility with applied one.
#         # TODO: update Xt/Xd.
#         data_nominal_idxs = nominal_idxs(data.Xt)
#         encoder = OneHotEncoder()
#         matrices = {}
#         if len(data_nominal_idxs) > 0:
#             nom = encoder.fit_transform(
#                 data.X[:, data_nominal_idxs]
#             ).toarray()
#             num = np.delete(data.X,
#                             data_nominal_idxs, axis=1).astype(float)
#             matrices['X'] = np.column_stack((nom, num))
#
#         return data.updated(self.transformations('u'), **matrices)
