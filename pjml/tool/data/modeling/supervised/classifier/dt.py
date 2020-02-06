from numpy.random import uniform
from sklearn.tree import DecisionTreeClassifier

from pjml.config.description.cs.componentcs import ComponentCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP, FixedP, IntP, RealP
from pjml.tool.data.modeling.supervised.predictor import Predictor


class DT(Predictor):
    """Decision Tree."""

    def __init__(self, **kwargs):
        super().__init__(kwargs, DecisionTreeClassifier(**kwargs))

    @classmethod
    def _cs_impl(cls):
        params = {
            'criterion': CatP(choice, items=['gini', 'entropy']),
            'splitter': FixedP('best'),
            'class_weight': CatP(choice, items=[None, 'balanced']),
            'max_features': CatP(choice, items=['auto', 'sqrt', 'log2', None]),

            'max_depth': IntP(uniform, low=2, high=1000),

            'min_samples_split': RealP(uniform, low=1e-6, high=0.3),
            'min_samples_leaf': RealP(uniform, low=1e-6, high=0.3),
            'min_weight_fraction_leaf': RealP(uniform, low=0.0, high=0.3),
            'min_impurity_decrease': RealP(uniform, low=0.0, high=0.2)
        }
        return ComponentCS(Node(params=params))
