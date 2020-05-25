import numpy as np
from numpy.random import uniform
from sklearn.feature_selection import f_classif, mutual_info_classif, \
    SelectPercentile, SelectFpr, SelectFdr, \
    SelectFwe, GenericUnivariateSelect, SelectKBest, chi2

from pjdata.step.transformation import Transformation
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import CatP, RealP
from pjml.tool.abc.mixin.exceptionhandler import BadComponent
from pjml.tool.data.algorithm import HeavyAlgorithm
from pjml.tool.model.model import Model


class SelectBest(HeavyAlgorithm):
    SCORE_FUNCTIONS = {
        "chi2": chi2,
        "f_classif": f_classif,
        "mutual_info_classif": mutual_info_classif
    }

    def __init__(self, score_func, k_perc, seed=0):
        sklearn_score = self.SCORE_FUNCTIONS[score_func]

        def algorithm_factory(nfeatures, random_state):
            if score_func == 'mutual_info_classif':
                np.random.seed(random_state)

            return SelectKBest(
                score_func=sklearn_score,
                k=int(np.ceil(nfeatures * k_perc))
            )

        config = {'score_func': score_func, 'k_perc': k_perc, 'seed': seed}
        if score_func != 'mutual_info_classif':
            del config['seed']
        super().__init__(config, algorithm_factory, {})

    @classmethod
    def _cs_impl(cls):
        params = {
            'score_func': CatP(
                choice,
                items=["chi2", "f_classif", "mutual_info_classif"]
            ),
            'k_perc': RealP(uniform, low=0.0, high=1.0)
        }
        return TransformerCS(nodes=[Node(params=params)])

    def _apply_impl(self, data):
        sklearn_model = self.algorithm_factory(data.X.shape[1])
        X_new = sklearn_model.fit_transform(*data.Xy)
        applied = data.updated(self.transformations('a'), X=X_new)
        return Model(self, data, applied, sklearn_model=sklearn_model)

    def _use_impl(self, data, sklearn_model=None):
        X_new = sklearn_model.transform(data.X)
        return data.updated(self.transformations('u'), X=X_new)

    def transformations(self, step, clean=True):
        if step == 'a':
            return tuple()
        elif step == 'u':
            return (Transformation(self, step),)
        else:
            raise BadComponent('Wrong current step:', step)
