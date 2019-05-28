""" Component module.
"""
import copy
import json
import time
import traceback
from abc import ABC, abstractmethod
from logging import warning
from uuid import uuid4

import numpy as np

from paje.base.data import Data
from paje.base.exceptions import ExceptionInApplyOrUse
from paje.evaluator.time import time_limit
from paje.result.sqlite import SQLite
from paje.result.storage import uuid


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, storage=None, show_warns=True):

        # self.model here refers to classifiers, preprocessors and, possibly,
        # some representation of pipelines or the autoML itself.
        # Another possibility is to generalize modules to a new class Module()
        # that has self.model.
        self.unfit = True
        self.model = None
        self.already_uuid = None  # UUID will be known only after build()
        self.dic = {}
        self.name = self.__class__.__name__
        self.tmp_uuid = uuid4().hex

        self.storage = storage
        self.data_used_for_apply = None
        self.locked = False
        self.failed = False

        # if True show warnings
        self.show_warns = show_warns

        self.already_serialized = None

    @abstractmethod
    def fields_to_store_after_use(self):
        pass

    @abstractmethod
    def fields_to_keep_after_use(self):
        """
        This method is only needed, because some components create incompatible
        input and output shapes.
        :return:
        """
        pass

    @abstractmethod
    def build_impl(self):
        pass

    def isdeterministic(self):
        return False

    @abstractmethod
    def apply_impl(self, data):
        """Todo the doc string
        """

    @abstractmethod
    def use_impl(self, data):
        """Todo the doc string
        """

    # @abstractmethod
    # def explain(self, X):
    #     """Explain prediction/transformation for the given instances.
    #     """
    #     raise NotImplementedError("Should it return probability\
    #                                distributions, rules?")

    @abstractmethod
    def tree_impl(cls, data):  # previously known as hyper_spaces_tree_impl
        """Todo the doc string
        """
        pass

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
        tree.name = self.name
        return tree

    def __str__(self, depth=''):
        return self.name + " " + str(self.dic)

    __repr__ = __str__

    def warning(self, msg):
        if self.show_warns:
            warning(msg)

    def error(self, msg):
        raise Exception(msg)

    def serialized(self):
        if self.already_serialized is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'serialized() <-' + self.name)
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
                        raise Exception('Real and integer hyperparameters need'
                                        ' a limit with two values: ' + str(k))
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
        if self.already_uuid is not None:
            self.error('Build cannot be called twice!')
        self = copy.copy(self)
        if self.storage is not None:
            self.storage.start()
        self.dic = dic
        if self.isdeterministic() and "random_state" in self.dic:
            del self.dic["random_state"]
        self.already_serialized = json.dumps(self.dic, sort_keys=True).encode()
        # self.already_serialized = zlib.compress(
        #     json.dumps(self.dic, sort_keys=True).encode())
        self.already_uuid = uuid(self.serialized())
        if 'name' in self.dic:
            del self.dic['name']
        self.build_impl()
        self.failed = False
        return self

    def handle_exceptions(self, f, maxtime=60):
        def function(train, test):
            try:
                if self.failed or self.locked:
                    raise Exception('Pipeline already failed/locked before!')
                if self.storage is not None:
                    self.storage.lock(self, train, test)
                with time_limit(maxtime):
                    start = time.clock()
                    testout = f(test)
                    time_spent = time.clock() - start
            except Exception as e:
                self.failed = True
                time_spent = None
                # Fake predictions for curated errors.
                self.warning('Trying to circumvent exception: >' + str(e) + '<')
                msgs = ['All features are either constant or ignored.',  # CB
                        'be between 0 and min(n_samples, n_features)',  # DR*
                        'excess of max_free_parameters:',  # MLP
                        'Pipeline already failed/locked before!',
                        # Preemptvely avoid
                        'Timed out!',
                        'Mahalanobis for too big data',
                        'MemoryError',
                        'On entry to DLASCL parameter number',  # Mahala knn
                        'excess of neighbors!',  # KNN
                        ]

                if any([str(e).__contains__(msg) for msg in msgs]):
                    testout = None
                    self.warning(e)
                else:
                    traceback.print_exc()
                    raise ExceptionInApplyOrUse(e)

            return testout, time_spent

        return function

    def apply(self, data=None):
        """Todo the doc string
        """
        print('Applying component...', self.name)
        self.data_used_for_apply = data
        if self.model is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'apply() <-' + self.name)
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warns:
            np.warnings.filterwarnings('ignore')

        if not data or self.failed or self.locked:
            self.warning('Trying to apply() failed/locked component or nodata!')
            return None

        f = self.handle_exceptions(self.apply_impl)
        if self.storage is None:
            output_data, time_spent = f(data, data)
        else:
            output_data, time_spent = self.storage.get_or_run(
                self, data, data, f)
        self.unfit = self.failed or self.locked

        if not self.show_warns:
            np.warnings.filterwarnings('always')

        return output_data

    def use(self, data: Data = None):
        """Todo the doc string
        """
        print('Using component...', self.name)
        if not data or self.failed or self.locked:
            self.warning('Trying to use() failed/locked component or nodata!')
            return None
        if self.unfit:
            raise UseWithoutApply('apply() should be called before '
                                  'use() <-' + self.name)

        f = self.handle_exceptions(self.use_impl)
        if self.storage is None:
            output_data, time_spent = f(self.data_used_for_apply, data)
        else:
            output_data, time_spent = self.storage.get_or_run(
                self, self.data_used_for_apply, data, f)

        return output_data

    def uuid(self):
        if self.already_uuid is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'uuid() <-' + self.name)
        return self.already_uuid


class UseWithoutApply(Exception):
    pass


class ApplyWithoutBuild(Exception):
    pass
