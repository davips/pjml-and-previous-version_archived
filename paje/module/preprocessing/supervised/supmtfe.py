# from pymfe.mfe import MFE
# from paje.base.component import Component
# import numpy as np
#
# from paje.base.data import Data
# from paje.base.hps import HPTree
#
#
# class SupMtFe(Component):
#     def apply_impl(self, data):
#         return self.use_impl(data)
#
#     def use_impl(self, data):
#         self.model.fit(*data.Xy)
#         names, values = self.model.extract(suppress_warnings=True)
#         X = np.array([values])
#         # TODO: suppressing NaNs with 0s
#         X[~np.isfinite(X)] = 0
#         return Data(name='meta', X=X) #TODO: choose a proper name
#
#     def build_impl(self):
#         self.model = MFE()
#
#     def tree_impl(cls, data):
#         HPTree({}, [])
