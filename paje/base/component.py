import zlib
import json
import hashlib
from abc import ABC, abstractmethod
from logging import warning

import numpy as np

from paje.base.hps import HPTree
from paje.data.data import Data
from paje.result.sqlite import SQLite
from paje.result.storage import uuid
from paje.util.auto_constructor import initializer


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, *args, in_place=False, show_warnings=True, memoize=False, **kwargs):
        # print(self.__class__)
        self.model = None  # Model here refers to classifiers, preprocessors and, possibly, some representation of pipelines or the autoML itself.
        self.in_place = in_place
        self.show_warnings = show_warnings
        self.memoize = memoize
        if memoize: self.storage = SQLite()
        # print()
        # print(type(self).__name__)
        # print('args', args)
        # print('kwargs', kwargs)
        # print()
        self.dict = kwargs
        self.already_serialized = None
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

    def handle_in_place(self, data: Data):
        """
        Switch between inplace and 'copying Data'.
        :param data: Data
        :return: Data
        """
        return data if data is None or self.in_place else data.copy()

    def handle_storage(self, data):
        """
        Overload this method if your component has no internal model, or isn't fit to memoizing.
        See how Pipeline do this.
        :param data:
        :return: data
        """
        if self.memoize:
            if self.model is None:
                self.error("This component cannot support storage, please implement a custom handle_storage to overcome this.")
            return self.storage.get_or_else(self, data, self.apply_impl)
        else:
            return self.apply_impl(data)

    def apply(self, data: Data = None) -> Data:
        """Todo the doc string
        """
        handled_data = self.handle_in_place(data)

        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.

        result = self.handle_storage(handled_data)

        if not self.show_warnings:
            np.warnings.filterwarnings('always')  # Mahalanobis in KNN needs to supress warnings due to NaN in linear algebra calculations. MLP is also verbose due to nonconvergence issues among other problems.
        return result

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
    def print_hyper_spaces_tree(cls, data=None):
        tree = cls.hyper_spaces_tree(data)
        print(tree)

    @classmethod
    def check_data(cls, data):
        if data is None:
            raise Exception(cls.__name__ + ' needs a dataset to be able to estimate maximum values for some hyperparameters.')

    def hyperpar_spaces_forest(self, data=None) -> HPTree:
        """
        This method, hyperpar_spaces_forest(), should be preferred over hyper_spaces_tree(),
        because not every Component can implement the latter. (See e.g. Pipeline)
        :param data:
        :return: [tree]
        """
        return self.__class__.hyper_spaces_tree(data)

    def __str__(self, depth=''):
        return self.__class__.__name__ + str(self.dict)

    __repr__ = __str__

    def warning(self, msg):
        if self.show_warnings:
            warning(msg)

    def error(self, msg):
        raise Exception(msg)

    def serialized(self):
        if self.already_serialized is None:
            self.already_serialized = zlib.compress(json.dumps(self.dict, sort_keys=True).encode())
        return self.already_serialized

    def __hash__(self):
        return uuid(self.serialized())

