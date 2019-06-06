""" Component module.
"""
import copy
import json
import os
from abc import ABC, abstractmethod
from uuid import uuid4

import numpy as np

from paje.base.exceptions import ApplyWithoutBuild, UseWithoutApply, \
    handle_exception
from paje.evaluator.time import time_limit
from paje.result.storage import uuid, pack_comp
from paje.util.log import *


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
        self.dic = {}
        self.name = self.__class__.__name__
        self.tmp_uuid = uuid4().hex

        self.storage = storage
        self._uuid = None  # UUID will be known only after build()
        self._uuid_train__mutable = None
        self.locked = False
        self.failed = False
        self.time_spent = None
        self.node = None
        self.max_time = None
        self._dump = None

        self.log = logging.getLogger('component')
        # if True show warnings
        self.log.setLevel(0)
        self.show_warns = show_warns

        self._serialized = None

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
        self.check_tree(tree)
        if tree.name is None:
            tree.name = self.name
        return tree

    @classmethod
    def check_tree(cls, tree):
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
            cls.check_tree(child)

    def build(self, **dic):
        # Check if build has already been called.
        if self._uuid is not None:
            self.error('Build cannot be called twice!')
        self = copy.copy(self)
        # if self.storage is not None:
        #     self.storage.open()
        self.dic = dic
        if self.isdeterministic() and "random_state" in self.dic:
            del self.dic["random_state"]

        # When the build is not created by a dic coming from a HPTree,
        #  it can be lacking a name.
        if 'name' not in self.dic:
            self.dic['name'] = self.name

        self._serialized = json.dumps(self.dic, sort_keys=True).encode()
        self._uuid = uuid(self.serialized())

        # 'name' and 'max_time' are reserved words, not for building.
        del self.dic['name']
        if 'max_time' in self.dic:
            self.max_time = self.dic.pop('max_time')

        self.build_impl()
        return self

    def handle_warnings(self):
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warns:
            np.warnings.filterwarnings('ignore')

    def dishandle_warnings(self):
        if not self.show_warns:
            np.warnings.filterwarnings('always')

    def lock(self, data, txt=''):
        self.storage.lock(self, data)
        self.msg(f'Locked [{txt}]!')

    def look_for_result(self, data):
        return self.storage and self.storage.get_result(self, data)

    def check_if_applied(self, data):
        if self._uuid_train__mutable is None:
            if self.storage is not None:
                print('It is possible that a previous apply() was '
                      'successfully stored, but its use() wasn\'t.',
                      'Please remove stored results for data', data.uuid(),
                      'and component', self.uuid())
            raise UseWithoutApply(f'{self.name} should be applied!')

    def check_if_built(self):
        self.serialized()  # Call just to raise exception, if needed.

    def apply(self, data=None):
        """Todo the doc string
        """
        # Checklist / get from storage -----------------------------------
        self.check_if_built()
        if data is None:
            raise Exception(f"Applying {self.name} on None !")

        self._uuid_train__mutable = data.uuid()
        output_data = self.look_for_result(data)
        if self.failed:
            self.msg(f"Won't apply on data {self.uuid_train()}\n"
                     f"Current {self.name} already failed before.")
            return output_data

        if self.locked:
            print(f"Won't apply {self.name} on data {self.uuid_train()}\n"
                  f"Current probably working at node [{self.node}].")
            return output_data

        # Apply if still needed  ----------------------------------
        if output_data is None:
            if self.storage is not None:
                try:
                    self.lock(data)
                except Exception as e:
                    print('Unexpected lock! Giving up my turn on apply()', e)
                    self.locked = True
                    print(data, 'giving up apply()')
                    return None

            self.handle_warnings()
            self.msg('Applying component' + self.name + '...')
            start = self.clock()
            try:
                if self.max_time is None:
                    output_data = self.apply_impl(data)
                else:
                    with time_limit(self.max_time):
                        output_data = self.apply_impl(data)
            except Exception as e:
                print(e)
                self.failed = True
                self.locked = False
                handle_exception(self, e)
            self.time_spent = self.clock() - start
            self.msg('Component ' + self.name + ' applied.')
            self.dishandle_warnings()

            if self.storage is not None:
                output_train_data = None if self.failed else self.use_impl(data)
                self.store_result(data, output_train_data)
        else:
            if self.storage is not None:
                # Check if use() will need a model. Assumes two results per set
                count = self.storage.count_results(self, data)
                if count == 1:
                    print('apply just for use() because results were '
                          'partially stored in a previous execution:'
                          f'comp: {self.uuid()}  data: {data.uuid()}')
                    output_data = self.apply_impl(data)
                # print('It is possible that a previous apply() was '
                #       'successfully stored, but its use() wasn\'t.',
                #       'Please remove stored results for data', data.uuid(),
                #       'and component', self.uuid())
        return output_data

    def use(self, data=None):
        """Todo the doc string
        """
        self.check_if_applied(data)

        # Checklist / get from storage -----------------------------------
        if data is None:
            self.msg(f"Using {self.name} on None returns None.")
            return None

        output_data = self.look_for_result(data)

        if self.locked:
            self.msg(f"Won't use {self.name} on data {self.uuid_train()}\n"
                     f"Current probably working at {self.node}.")
            return output_data

        if self.failed:
            self.msg(f"Won't use on data {data.uuid()}\n"
                     f"Current {self.name} already failed before.")
            return output_data

        # Use if still needed  ----------------------------------
        if output_data is None:
            if self.storage is not None:
                try:
                    self.lock(data, 'using')
                except Exception as e:
                    print('Unexpected lock! Giving up my turn on use()...', e)
                    self.locked = True
                    print(data, 'giving up use()')
                    return None

            self.handle_warnings()
            print('Using component', self.name, '...')

            # TODO: put time limit and/or exception handling like in apply()?
            start = self.clock()
            output_data = self.use_impl(data)  # TODO:handle excps mark failed
            self.time_spent = self.clock() - start

            self.msg('Component ' + self.name + 'used.')
            self.dishandle_warnings()

            if self.storage is not None:
                self.store_result(data, output_data)
        return output_data

    def uuid(self):
        if self._uuid is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'uuid() <-' + self.name)
        return self._uuid

    def __str__(self, depth=''):
        return self.name + " " + str(self.dic)

    __repr__ = __str__

    def msg(self, msg):
        print(msg)
        # self.log.log(1, msg)

    def warning(self, msg):
        print(msg)
        # self.log.warning(2, msg)

    def debug(self, msg):
        print(msg)
        # self.log.debug(3, msg)

    def error(self, msg):
        print(msg)
        # self.log.error(4, msg)
        raise Exception(msg)

    def serialized(self):
        if self._serialized is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'serialized() <-' + self.name)
        return self._serialized

    def store_data(self, data):
        self.storage.store_data(data)

    def store_result(self, input_data, output_data):
        """
        :param input_data:
        :param output_data:
        :return:
        """
        self.storage.store(self, input_data, output_data)

    def clock(self):
        t = os.times()
        return t[4]
        # return usage[0] + usage[1]  # TOTAL CPU whole-system time

    def dump(self):
        self.check_if_applied()  # It makes no sense to store an unapplied comp.
        if self._dump is None:
            self._dump = pack_comp(self)
        return self._dump

    def uuid_train(self):
        if self._uuid_train__mutable is None:
            raise Exception('This component should be applied to have '
                            'a UUID of the training Data.', self.name)
        return self._uuid_train__mutable
