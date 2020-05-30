from functools import partial

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP
from pjml.tool.data.modeling.supervised.predictor import Predictor


class NB(Predictor):
    """Naive Bayes implementations: gaussian, bernoulli."""

    def __init__(self, distribution="gaussian", **kwargs):
        if distribution == "gaussian":
            func = GaussianNB
        elif distribution == "bernoulli":
            func = BernoulliNB
        else:
            raise Exception('Wrong distribution:', distribution)
        config = {'distribution': distribution}
        super().__init__(config, func, {}, **kwargs, deterministic=True)
        self.distribution = distribution

    @classmethod
    def _cs_impl(cls):
        params = {
            'distribution': CatP(choice, items=['gaussian', 'bernoulli'])
        }
        return TransformerCS(nodes=[Node(params=params)])
