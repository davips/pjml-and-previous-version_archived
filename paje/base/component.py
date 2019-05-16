""" Component module.
"""
import copy
import json
import zlib
from abc import ABC, abstractmethod
from logging import warning

import numpy as np

from paje.base.exceptions import ExceptionInApply
from paje.data.data import Data
from paje.result.sqlite import SQLite
from paje.result.storage import uuid


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, in_place=False, memoize=False, show_warns=True):

        # self.model here refers to classifiers, preprocessors and, possibly,
        # some representation of pipelines or the autoML itself.
        # Another possibility is to generalize modules to a new class Module()
        # that has self.model.
        self.model = None
        self.dic = {}

        # if True no copy will be made
        self.in_place = in_place
        # if True show warnings
        self.show_warnings = show_warns

        self.memoize = memoize
        if memoize:
            self.storage = SQLite()

        self.already_serialized = None

    def isdeterministic(self):
        return False

    @abstractmethod
    def instantiate_impl(self):
        pass

    @abstractmethod
    def apply_impl(self, data):
        """Todo the doc string
        """

    @abstractmethod
    def use_impl(self, data):
        """Todo the doc string
        """

    def handle_inplace(self, data: Data):
        """
        Switch between inplace and 'copying Data'.
        :param data: Data
        :return: Data
        """
        return data if data is None or self.in_place else data.copy()

    def handle_storage(self, data):
        """
        Overload this method if your component has no internal model, or isn't
            fit to memoizing.
        See how Pipeline do this.
        :param data:
        :return: data
        """
        if self.memoize:
            if self.model is None:
                self.error("This component cannot support storage, \
                           please implement a custom handle_storage to \
                           overcome this.")
            return self.storage.get_or_else(self, data, self.apply_impl)

        try:
            return self.apply_impl(data)
        except Exception as e:
            raise ExceptionInApply(e)

    # @abstractmethod
    # def explain(self, X):
    #     """Explain prediction/transformation for the given instances.
    #     """
    #     raise NotImplementedError("Should it return probability\
    #                                distributions, rules?")

    @classmethod
    @abstractmethod
    def tree_impl(cls, data=None):  # previously known as hyper_spaces_tree_impl
        """Todo the doc string
        """
        pass

    @classmethod
    def tree(cls, data=None):  # previously known as hyper_spaces_tree
        """
        Only call this method instead of hyperpar_spaces_forest() if you know
        what you are doing! (E.g. calling from a classifier)
        :param data:
        :return: tree
        """
        tree = cls.tree_impl(data)
        cls.check(tree)
        return tree

    @classmethod
    def print_tree(cls,
                   data=None):  # previously known as print_hyper_spaces_tree
        tree = cls.tree(data)
        print(tree)

    @classmethod
    def check_data(cls, data):
        if data is None:
            raise Exception(cls.__name__ + ' needs a dataset to be able to \
                            estimate maximum values for some hyperparameters.')

    def forest(self, data=None):  # previously known as hyperpar_spaces_forest
        """
        This method, hyperpar_spaces_forest(), should be preferred over
        hyper_spaces_tree(),
        because not every Component can implement the latter.
        (See e.g. Pipeline)
        :param data:
        :return: [tree]
        """
        tree = self.__class__.tree(data)
        tree.name = self.__class__.__name__
        return tree

    def __str__(self, depth=''):
        return self.__class__.__name__ + " " + str(self.dic)

    __repr__ = __str__

    def warning(self, msg):
        if self.show_warnings:
            warning(msg)

    def error(self, msg):
        raise Exception(msg)

    def serialized(self):
        if self.already_serialized is None:
            self.already_serialized = zlib.compress(
                json.dumps(self.dic, sort_keys=True).encode())
        return self.already_serialized

    def __hash__(self):
        return uuid(self.serialized())

    @classmethod
    def check(cls, tree):
        try:
            dic = tree.dic
        except Exception as e:
            print(e)
            print()
            print(cls.__name__, ' <- problematic class')
            print()
            raise Exception('Problems with hyperparameter space')

        try:
            for k in dic:
                t = dic[k][0]
                v = dic[k][1]
                if t == 'c' or t == 'o':
                    if not isinstance(v, list):
                        raise Exception('Categorical and ordinal \
                                        hyperparameters need a list of \
                                        values: ' + str(k))
                else:
                    if len(v) != 2:
                        raise Exception('Real and integer hyperparameters \
                                        need a limit with two \
                                        values: ' + str(k))
        except Exception as e:
            print(e)
            print()
            print(cls.__name__)
            print()
            raise Exception('Problems with hyperparameter space: ' + str(dic))

        for child in tree.children:
            cls.check(child)

    def instantiate(self, **dic):
        self = copy.copy(self)
        self.dic = dic
        if self.isdeterministic() and "random_state" in self.dic:
            del self.dic["random_state"]
        self.instantiate_impl()
        return self
        #
        # return self.__class__(self.in_place, self.memoize, self.show_warnings,
        #                       **dic)

    def apply(self, data: Data = None) -> Data:
        """Todo the doc string
        """
        handled_data = self.handle_inplace(data)

        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warnings:
            np.warnings.filterwarnings('ignore')

        result = self.handle_storage(handled_data)

        if not self.show_warnings:
            np.warnings.filterwarnings('always')

        return result

    def use(self, data: Data = None) -> Data:
        """Todo the doc string
        """
        return self.use_impl(self.handle_inplace(data))

    def print_forest(self, data=None):
        print(self.forest(data))
