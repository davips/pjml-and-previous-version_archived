from sklearn.tree import DecisionTreeClassifier

from paje.base.hps import ConfigSpace
from paje.base.hps import CatHP, RealHP, IntHP
from numpy.random import choice, uniform
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class DT(Classifier):
    def build_impl(self, **config):
        self.model = DecisionTreeClassifier(**self.config)

    @classmethod
    def tree_impl(self):
        # Sw
        # cs = ConfigSpace('Switch')
        # st = cs.start()
        # st.add_children([a.start, b.start, c.start])
        # cs.finish([a.end,b.end,c.end])

        config_space = ConfigSpace('DT')
        start = config_space.start()
        node = config_space.node()
        start.add_child(node)

        node.add_hp(CatHP('criterion', choice, a=['gini', 'entropy']))
        node.add_hp(CatHP('splitter', choice, a=['best']))
        node.add_hp(CatHP('class_weight', choice, a=[None, 'balanced']))
        node.add_hp(CatHP('max_features', choice,
                          a=['auto', 'sqrt', 'log2', None]))

        node.add_hp(IntHP('max_depth', uniform, low=2, high=1000))

        node.add_hp(RealHP('min_samples_split', uniform, low=1e-6, high=0.3))
        node.add_hp(RealHP('min_samples_leaf', uniform, low=1e-6, high=0.3))
        node.add_hp(RealHP('min_weight_fraction_leaf', uniform, low=0.0, high=0.3))
        node.add_hp(RealHP('min_impurity_decrease', uniform, low=0.0, high=0.2))

        return config_space
