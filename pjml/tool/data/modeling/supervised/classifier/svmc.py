from numpy.random import uniform
from sklearn.svm import SVC

from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP, IntP, RealP, CatP, OrdP
from pjml.tool.data.modeling.supervised.predictor import Predictor


class SVMC(Predictor):
    def __init__(self, enhance=True, model=True, **sklconfig):
        super().__init__(sklconfig, SVC,
                         enhance=enhance, model=model)

    @classmethod
    def _cs_impl(cls):
        # todo: set random seed; set 'cache_size'
        kernel_linear = Node({'kernel': FixedP('linear')})

        kernel_poly = Node({
            'kernel': FixedP('poly'),
            'degree': IntP(uniform, low=0, high=10),
            'coef0': RealP(uniform, low=0.0, high=100)
        })

        kernel_rbf = Node({
            'kernel': FixedP('rbf')
        })

        kernel_sigmoid = Node({
            'kernel': FixedP('sigmoid'),
            'coef0': RealP(uniform, low=0.0, high=100),
        })

        kernel_nonlinear = Node(
            {'gamma': RealP(uniform, low=0.00001, high=100)},
            children=[kernel_poly, kernel_rbf, kernel_sigmoid]
        )

        top = Node(
            {
                'C': RealP(uniform, low=1e-4, high=100),
                'shrinking': CatP(choice, items=[True, False]),
                'probability': FixedP(False),
                'tol': OrdP(choice, items=[
                    0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,
                    1, 10, 100, 1000, 10000
                ]),
                'class_weight': CatP(choice, items=[None, 'balanced']),
                # 'verbose': [False],
                'max_iter': FixedP(1000000),
                'decision_function_shape': CatP(choice, items=['ovr', 'ovo'])
            },
            children=[kernel_linear, kernel_nonlinear]
        )

        return TransformerCS(nodes=[top])
