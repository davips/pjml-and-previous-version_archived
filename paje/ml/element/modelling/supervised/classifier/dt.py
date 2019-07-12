from sklearn.tree import DecisionTreeClassifier

from paje.base.hp import CatHP, IntHP, RealHP
from paje.base.hps import ConfigSpace, Node
from numpy.random import choice, uniform
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class DT(Classifier):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = DecisionTreeClassifier(**self.config)

    @classmethod
    def tree_impl(cls):
        # Sw
        # cs = ConfigSpace('Switch')
        # st = cs.start()
        # st.add_children([a.start, b.start, c.start])
        # cs.finish([a.end,b.end,c.end])

        hps = [
            CatHP('criterion', choice, a=['gini', 'entropy']),
            CatHP('splitter', choice, a=['best']),
            CatHP('class_weight', choice, a=[None, 'balanced']),
            CatHP('max_features', choice,
                  a=['auto', 'sqrt', 'log2', None]),

            IntHP('max_depth', uniform, low=2, high=1000),

            RealHP('min_samples_split', uniform, low=1e-6, high=0.3),
            RealHP('min_samples_leaf', uniform, low=1e-6, high=0.3),
            RealHP('min_weight_fraction_leaf', uniform, low=0.0, high=0.3),
            RealHP('min_impurity_decrease', uniform, low=0.0, high=0.2)
        ]

        return ConfigSpace(name='DT', hps=hps)
