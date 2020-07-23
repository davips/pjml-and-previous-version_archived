# from numpy.random import uniform
# from sklearn.ensemble import RandomForestClassifier
#
# from pjml.config.description.cs.transformercs import TransformerCS
# from pjml.config.description.distributions import choice
# from pjml.config.description.node import Node
# from pjml.config.description.parameter import CatP, RealP, IntP
# from pjml.tool.data.modeling.supervised.predictor import Predictor
#
#
# class RF(Predictor):
#     """Random Forest."""
#
#     def __init__(self, **sklconfig):
#         super().__init__(sklconfig, RandomForestClassifier)
#
#     @classmethod
#     def _cs_impl(cls):
#         n_estimators = [100, 500, 1000, 3000, 5000]
#         params = {
#             'bootstrap': CatP(choice, items=[True, False]),
#             'criterion': CatP(choice, items=['gini', 'entropy']),
#             'max_features': CatP(choice, items=['auto', 'sqrt', 'log2', None]),
#             'min_impurity_decrease': RealP(uniform, low=0.0, high=0.2),
#             'min_samples_split': RealP(uniform, low=1e-6, high=0.3),
#             'min_samples_leaf': RealP(uniform, low=1e-6, high=0.3),
#             'min_weight_fraction_leaf': RealP(uniform, low=0.0, high=0.3),
#             'max_depth': IntP(uniform, low=2, high=1000),
#             'n_estimators': CatP(choice, items=n_estimators),
#         }
#         return TransformerCS(nodes=[Node(params=params)])
