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
from paje.util.encoders import pack_comp, uuid, json_pack
from paje.util.log import *


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, storage=None, show_warns=True, dump_it=None):

        # self.model here refers to classifiers, preprocessors and, possibly,
        # some representation of pipelines or the autoML itself.
        # Another possibility is to generalize modules to a new class Module()
        # that has self.model.
        self.unfit = True
        self.model = None
        self.dic = {}
        self.name = self.__class__.__name__
        self.module = self.__class__.__module__
        self.tmp_uuid = uuid4().hex  # used when eliminating End* from tree

        self.storage = storage
        self._uuid = None  # UUID will be known only after build()

        # Each apply() uses a different training data, so this uuid is mutable
        self._train_data_uuid__mutable = None

        self.locked_by_others = False
        self.failed = False
        self.time_spent = None
        self.node = None
        self.max_time = None
        self.mark = None
        self._model_dump = None
        self._model_uuid = None
        self._describe = None

        # Whether to dump the model or not, if a storage is given.
        self.dump_it = dump_it or None  # 'or'->possib.vals=Tru,Non See sql.py

        self.log = logging.getLogger('component')
        # if True show warnings
        self.log.setLevel(0)
        self.show_warns = show_warns

        self._serialized = None

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

    def build(self, **dic):
        # Check if build has already been called. This is the case when one
        # calls build() on an already built instance of component.

        # TODO: Is this check necessary?
        # if self._uuid is not None:
        #     self.error('Build cannot be called on a built component!')

        obj_copied = copy.copy(self)
        # descobrir no git log porque colocamos esse update aqui!!!!!
        # voltei para o que era antes para parar de fazer bruxaria
        # de alterar o bestpipeline quando fazia o build do proximo pipe.
        # Se for mesmo necessario dar update, podemos dar update usando uma
        # copia de dic? ou seria inutil?
        # obj_copied.dic.update(dic)
        obj_copied.dic = dic

        if obj_copied.isdeterministic() and "random_state" in obj_copied.dic:
            del obj_copied.dic["random_state"]

        # Create an encoded (uuid) unambiguous (sorted) version of args_set.
        obj_copied._serialized = 'building'
        obj_copied._uuid = uuid(obj_copied.serialized().encode())

        obj_copied.build_impl()
        return obj_copied

    def describe(self):
        if self._describe is None:
            self._describe = {
                'module': self.module,
                'name': self.name,
                'sub_components': []
            }
        return self._describe

    def apply(self, data=None):
        """Todo the doc string
        """
        # Checklist / get from storage -----------------------------------
        self.check_if_built()
        if data is None:
            raise Exception(f"Applying {self.name} on None !")

        self._train_data_uuid__mutable = data.uuid()

        output_data = None
        if self.storage is not None:
            output_data = self.storage.get_result(self, 'a', data)

        if self.failed:
            self.msg(f"Won't apply on data {data.name()}"
                     f"\nCurrent {self.name} already failed before.")
            return output_data

        if self.locked_by_others:
            print(f"Won't apply {self.name} on data "
                  f"{data.name()}\n"
                  f"Currently probably working at node [{self.node}].")
            return output_data

        # Apply if still needed  ----------------------------------
        if output_data is None:
            if self.storage is not None:
                self.storage.lock(self, 'a', data)

            self.handle_warnings()
            if self.name != 'CV':
                self.msg('Applying ' + self.name + '...')
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
                self.locked_by_others = False
                handle_exception(self, e)
            self.time_spent = self.clock() - start
            # self.msg('Component ' + self.name + ' applied.')
            self.dishandle_warnings()

            if self.storage is not None:
                self.storage.store_result(self, 'a', data, output_data)

        return output_data

    def use(self, data=None):
        """Todo the doc string
        """
        self.check_if_applied()

        if self.storage is not None:
            # If the component was applied (probably simulated by storage),
            # but there is no model, we reapply it...
            if self.model is None:
                print('It is possible that a previous apply() was '
                      'successfully stored, but its use() wasn\'t.'
                      'Or you are trying to use in new data.')
                print('Trying to recover training data from storage to apply '
                      'just to induce a model usable by use()...\n'
                      f'comp: {self.sid()}  data: {data.sid()} ...')
                self.apply_impl(stored_train_data)

        # Checklist / get from storage -----------------------------------
        if data is None:
            self.msg(f"Using {self.name} on None returns None.")
            return None

        output_data = None
        if self.storage is not None:
            output_data = self.storage.get_result(self, 'u', data)

        if self.locked_by_others:
            self.msg(f"Won't use {self.name} on data "
                     f"{data.name()}\n"
                     f"Currently probably working at {self.node}.")
            return output_data

        if self.failed:
            self.msg(f"Won't use on data {data.sid()}\n"
                     f"Current {self.name} already failed before.")
            return output_data

        # Use if still needed  ----------------------------------
        if output_data is None:
            if self.storage is not None:
                self.storage.lock(self, 'u', data)

            self.handle_warnings()
            if self.name != 'CV':
                print('Using ', self.name, '...')

            # TODO: put time limit and/or exception handling like in apply()?
            start = self.clock()
            output_data = self.use_impl(data)  # TODO:handle excps mark failed
            self.time_spent = self.clock() - start

            # self.msg('Component ' + self.name + 'used.')
            self.dishandle_warnings()

            if self.storage is not None:
                self.storage.store_result(self, 'u', data, output_data)
        return output_data

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

    def uuid(self):
        if self._uuid is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'uuid() <-' + self.name)
        return self._uuid

    def model_uuid(self):
        self.check_if_applied()
        if self._model_uuid is None and self.dump_it:
            self._model_uuid = uuid(self.model_dump())
        return self._model_uuid

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
        """
        Calculate a representation of this built component.
        In the first call, remove reserved words from args_set
         (like 'mark', 'name').
        :return: 19-byte string
        """
        if self._serialized is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'serialized() <-' + self.name)

        # The first time serialized() is called,
        # it has to put needed (and remove unneeded) args from arg_set.
        if self._serialized == 'building':
            # When the build is not created by a dic coming from a HPTree,
            #  it can be lacking a name, so here we put it.
            if 'name' not in self.dic:
                self.dic['name'] = self.name

            # 'mark' should not identify components, it only marks results.
            # when a component is loaded from storage, nobody knows whether
            # it was part of an experiment or not, except by consulting
            # the field 'mark' of the registered result.
            if 'mark' in self.dic:
                self.mark = self.dic.pop('mark')

            # 'description' is needed because some components are not entirely
            # described by build() args.
            self.dic['description'] = self.describe()

            # Create an unambiguous (sorted) version of
            # args_set (build args+name+max_time) + init_vars (description).
            self._serialized = json_pack(self.dic)

            # 'description','name','max_time' are reserved words, not for
            # building.
            del self.dic['name']
            if 'max_time' in self.dic:
                self.max_time = self.dic.pop('max_time')
            del self.dic['description']

        return self._serialized

    @staticmethod
    def clock():
        t = os.times()
        return t[4]
        # return usage[0] + usage[1]  # TOTAL CPU whole-system time

    # TODO: is config dump (or even entire component dump) really needed?
    #  we can reconstruct the component using the kwargs and model_dump
    # def conf_dump(self):
    #     """Compact everything except self.model"""
    #     not implemented!
    #     self.check_if_applied()  # It makes no sense to store an unapplied comp.
    #     if self._dump is None:
    #         # TODO: dumping entire component (?),
    #         #  the user would need pajé to extract the model from it,
    #         #  or to have a model dump.
    #         self._dump = pack_comp(self)
    #     return self._dump

    def model_dump(self):
        """
        Compact the internal model (e.g., sklearn instance) of this component.
        :return:
        """
        self.check_if_applied()
        if self._model_dump is None and self.dump_it:
            self._model_dump = pack_comp(self.model)
        return self._model_dump

    def train_data_uuid__mutable(self):
        if self._train_data_uuid__mutable is None:
            raise Exception('This component should be applied to have '
                            'an internal training data uuid.', self.name)
        return self._train_data_uuid__mutable

    def handle_warnings(self):
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warns:
            np.warnings.filterwarnings('ignore')

    def dishandle_warnings(self):
        if not self.show_warns:
            np.warnings.filterwarnings('always')

    def check_if_applied(self):
        if self._train_data_uuid__mutable is None:
            raise UseWithoutApply(f'{self.name} should be applied!')

    def check_if_built(self):
        self.serialized()  # Call just to raise exception, if needed.

    @staticmethod
    def resurrect_from_dump(model_dump, kwargs):
        """Recreate a component from ashes."""
        raise Exception('Not implemented')

    def sid(self):
        """
        Short uuID
        First 5 chars of uuid for printing purposes.
        :return:
        """
        return self.uuid()[:5]

    # This may be useful in the future: ================================
    # @abstractmethod
    # def touched_fields(self):
    #     """
    #     Matrices transformed or created by this component.
    #     Useful to be able to store only new info.
    #     :return:
    #     """
    #     pass
    #
    # @abstractmethod
    # def still_compatible_fields(self):
    #     """
    #     Some components create incompatible input and output shapes.
    #     Useful to merge results with complementar info.
    #     :return:
    #     """
    #     pass
    #
    # @abstractmethod
    # def needed_fields(self):
    #     """
    #     Matrices needed by this component, but will not be necessarily touched.
    #     Useful to be able to store only meaningful previous info.
    #     :return:
    #     """
    #     pass
    # ================================================================
