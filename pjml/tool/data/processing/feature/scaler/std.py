# from functools import partial
#
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
#
# from pjml.config.description.cs.transformercs import TransformerCS
# from pjml.config.description.distributions import choice
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import CatP
# from pjml.tool.data.processing.feature.scaler.scaler import Scaler
#
#
# class Std(Scaler):
#     def __init__(self, operation='full'):
#         if operation == 'full':
#             with_mean, with_std = True, True
#         else:
#             with_mean, with_std = 'translate' == operation, 'scale' == operation
#
#         sklconfig = {'with_mean': with_mean, 'with_std': with_std}
#         config = {'operation': operation}
#         super().__init__(config, StandardScaler, sklconfig, deterministic=True)
#
#     @classmethod
#     def _cs_impl(cls, data=None):
#         params = {
#             'operation':
#                 CatP(choice, items=['full', 'translate', 'scale'])
#         }
#         return TransformerCS(nodes=[Node(params=params)])
