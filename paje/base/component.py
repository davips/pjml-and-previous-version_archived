""" Component module.
"""
import copy
import json
import zlib
from abc import ABC, abstractmethod
from logging import warning

import numpy as np

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.base.data import Data
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
        self.uuid = None  # UUID will be known only after build()
        self.dic = {}

        # Store apply() results in disk?
        self.memoize = memoize
        self.storage = None  # Defined in build() when needed, to avoid locking.

        # if True no copy will be made
        self.in_place = in_place

        # if True show warnings
        self.show_warns = show_warns

        self.already_serialized = None

    def isdeterministic(self):
        return False

    @abstractmethod
    def build_impl(self):
        pass

    @abstractmethod
    def apply_impl(self, data):
        """Todo the doc string
        """

    @abstractmethod
    def use_impl(self, data):
        """Todo the doc string
        """

    def handle_storage(self, data):
        """
        Overload this method if your component has no internal model, or isn't
            fit to memoizing.
        See how Pipeline do this.
        :param data:
        :return: data
        """
        self.error('not implemented!')
        # TODO: decide and correct dump storage: self.handle_storage(data)

        if self.memoize:
            if self.model is None:
                self.error("This component " + self.__class__.__name__ +
                           " cannot support storage, please implement a" +
                           " custom handle_storage to overcome this.")
            return self.storage.get_or_else(self, data, self.apply_impl)

        try:
            return self.apply_impl(data)
        except Exception as e:
            raise ExceptionInApplyOrUse(e)

    # @abstractmethod
    # def explain(self, X):
    #     """Explain prediction/transformation for the given instances.
    #     """
    #     raise NotImplementedError("Should it return probability\
    #                                distributions, rules?")

    def tree_impl(cls, data):  # previously known as hyper_spaces_tree_impl
        """Todo the doc string
        """
        pass

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

    def tree(self, data=None):  # previously known as hyperpar_spaces_forest
        """
        :param data:
        :return: [tree]
        """
        # TODO: all child classes mark tree_impl as classmethod, turn it into
        #  instance method?
        tree = self.tree_impl(data)
        self.check(tree)
        tree.name = self.__class__.__name__
        return tree

    def __str__(self, depth=''):
        return self.__class__.__name__ + " " + str(self.dic)

    __repr__ = __str__

    def warning(self, msg):
        if self.show_warns:
            warning(msg)

    def error(self, msg):
        raise Exception(msg)

    def serialized(self):
        if self.already_serialized is None:
            self.already_serialized = zlib.compress(
                json.dumps(self.dic, sort_keys=True).encode())
        return self.already_serialized

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

    def build(self, **dic):
        # Check if build has already been called.
        if self.uuid is not None:
            self.error('Build cannot be called twice!')
        self = copy.copy(self)
        if self.memoize:
            self.storage = SQLite()
        self.dic = dic
        if self.isdeterministic() and "random_state" in self.dic:
            del self.dic["random_state"]
        self.uuid = uuid(self.serialized())
        if 'name' in self.dic:
            del self.dic['name']
        self.build_impl()
        return self
        #
        # return self.__class__(self.in_place, self.memoize, self.show_warnings,
        #                       **dic)

    # @profile
    def apply(self, data=None):
        """Todo the doc string
        """
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warns:
            np.warnings.filterwarnings('ignore')

        result = self.apply_impl(data)

        if not self.show_warns:
            np.warnings.filterwarnings('always')

        return result

    def use(self, data: Data = None) -> Data:
        """Todo the doc string
        """
        return self.use_impl(data)

    def print_forest(self, data=None):
        print(self.tree(data))
