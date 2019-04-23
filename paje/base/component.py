from abc import ABC, abstractmethod

import numpy as np

from paje.base.hps import HPTree
from paje.data.data import Data
from paje.util.auto_constructor import initializer


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, *args, in_place=False, show_warnings=True, **kwargs):
        # print(self.__class__)
        self.model = None  # Model here refers to classifiers, preprocessors and, possibly, some representation of pipelines or the autoML itself.
        self.in_place = in_place
        self.show_warnings = show_warnings
        # print()
        # print(type(self).__name__)
        # print('args', args)
        # print('kwargs', kwargs)
        # print()
        self.dict = kwargs
        self.init_impl(*args, **kwargs)

    @abstractmethod
    def init_impl(self, *args, **kwargs):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def apply_impl(self, data):
        """Todo the doc string
        """
        pass

    @abstractmethod
    def use_impl(self, data):
        """Todo the doc string
        """
        pass

    def handle_warnings(self, f, data):
        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.
        result = f(data)
        if not self.show_warnings:
            np.warnings.filterwarnings('always')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.
        return result

    def handle_in_place(self, data: Data):
        """
        Switch between inplace and 'copying Data'.
        :param data: Data
        :return: Data
        """
        return data if data is None or self.in_place else data.copy()

    def apply(self, data: Data = None) -> Data:
        """Todo the doc string
        """

        # TODO: If this result was already calculated before, recover if from Cache.
        return self.handle_warnings(self.apply_impl, self.handle_in_place(data))

    def use(self, data: Data = None) -> Data:
        """Todo the doc string
        """
        return self.use_impl(self.handle_in_place(data))

    # @abstractmethod
    # def explain(self, X):
    #     """Explain prediction/transformation for the given instances.
    #     """
    #     raise NotImplementedError("Should it return probability distributions, rules?")

    @classmethod
    @abstractmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        """Todo the doc string
        """
        pass

    @classmethod
    def hyper_spaces_tree(cls, data=None):
        """
        Only call this method instead of hyperpar_spaces_forest() if you know what you are doing!
        :param data:
        :return: tree
        """
        tree = cls.hyperpar_spaces_tree_impl(data)
        dic = tree.dic
        # TODO: check children also (recursively).
        try:
            for k in dic:
                t = dic[k][0]
                v = dic[k][1]
                if t == 'c' or t == 'o':
                    if not isinstance(v, list):
                        raise Exception('Categorical and ordinal hyperparameters need a list of values: ' + str(k))
                else:
                    if len(v) != 2:
                        raise Exception('Real and integer hyperparameters need a limit with two values: ' + str(k))
        except Exception as e:
            print(e)
            print()
            raise Exception('Problems with hyperparameter space: ' + str(dic))
        return tree

    @classmethod
    def print_hyper_spaces_tree(cls, data=None, depth=''):
        tree = cls.hyper_spaces_tree(data)
        print(depth + str(tree.dic))
        depth += '     '
        for child in tree.children:
            cls.print_hyper_spaces_tree(child, depth)

    @classmethod
    def check_data(cls, data):
        if data is None:
            print(cls.__name__ + ' needs a dataset to be able to estimate maximum values for some hyperparameters.')
            exit(0)

    def hyperpar_spaces_forest(self, data=None) -> [HPTree]:
        """
        This method, hyperpar_spaces_forest(), should be preferred over hyper_spaces_tree(),
        because not every Component can implement the latter. (See e.g. Pipeline)
        :param data:
        :return: [tree]
        """
        return [self.__class__.hyper_spaces_tree(data)]

    def __str__(self):
        return self.dict
