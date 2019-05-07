from sklearn.dummy import DummyClassifier
from sklearn.svm import NuSVC

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class SVM(Classifier):

    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)

        self.model = NuSVC(**kwargs)

    def apply_impl(self, data):
        try:
            return super().apply_impl(data)
        except:
            if self.show_warnings:
                print('Falling back to random classifier, possible due to convergence problems (bad "nu" value, for instance).')
            self.model = DummyClassifier(strategy='uniform')
            return super().apply_impl(data)

    @classmethod
    def tree_impl(cls, data=None):
        cls.check_data(data)
        # todo: set random seed; set 'cache_size'
        max_iter = data.n_instances()
        dic = {
            'nu': ['r', [0.00000001, 1.0]],
            'shrinking': ['c', [True, False]],
            'probability': ['c', [True, False]],
            'tol': ['o', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]],
            # 'class_weight': [None],
            # 'verbose': [False],
            'max_iter': ['z', [1, max_iter]],
            'decision_function_shape': ['c', ['ovr', 'ovo']]
        }

        kernel_linear = HPTree({'kernel': ['c', ['linear']]}, children=[])

        kernel_poly = HPTree({
            'kernel': ['c', ['poly']],
            'degree': ['z', [0, 10]],
            'coef0': ['r', [0.0, 100]],
        }, children=[])

        kernel_rbf = HPTree({'kernel': ['c', ['rbf']]}, children=[])

        kernel_sigmoid = HPTree({
            'kernel': ['c', ['sigmoid']],
            'coef0': ['r', [0.0, 100]],
        }, children=[])

        kernel_nonlinear = HPTree({'gamma': ['r', [0.00001, 100]]}, children=[kernel_poly, kernel_rbf, kernel_sigmoid])

        return HPTree(dic, children=[kernel_linear, kernel_nonlinear])
