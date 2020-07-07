# from functools import partial
#
# from sklearn.preprocessing import MinMaxScaler
#
# from pjml.config.description.cs.transformercs import TransformerCS
# from pjml.config.description.distributions import choice
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import CatP
# from pjml.tool.data.processing.feature.scaler.scaler import Scaler
#
#
# class MinMax(Scaler):
#     def __init__(self, **sklconfig):
#         super().__init__(sklconfig, MinMaxScaler, deterministic=True)
#
#     @classmethod
#     def _cs_impl(cls):
#         params = {
#             'feature_range': CatP(choice, items=[(-1, 1), (0, 1)])
#         }
#         return TransformerCS(nodes=[Node(params=params)])
