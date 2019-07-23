""" Component module.
"""
import copy
import os
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import choice

from paje.base.exceptions import ApplyWithoutBuild, UseWithoutApply, \
    handle_exception
from paje.base.hp import CatHP, FixedHP
from paje.evaluator.time import time_limit
from paje.util.encoders import uuid, json_pack


class StorageSettings:
    def __init__(self, storage=None, dump_it=None):
        self.storage = storage
        self.dump_it = dump_it


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, config, storage_settings=None):
        self._storage = storage_settings and storage_settings.storage
        self._dump_it = storage_settings and storage_settings.dump_it

        self.config = config.copy()

        # 'mark' should not identify components, it only marks results.
        # when a component is loaded from storage, nobody knows whether
        # it was part of an experiment or not, except by consulting
        # the field 'mark' of the registered result. 'max_time' is reserved word
        self.mark = self.config.pop('mark') if 'mark' in config else None
        self.max_time = self.config.pop('max_time') \
            if 'max_time' in config else None

        # 'random_state' is reserved word
        if self.isdeterministic() and 'random_state' in self.config:
            del self.config['random_state']

        if 'class' not in config:
            self.config['class'] = self.__class__.__name__
        if 'module' not in config:
            self.config['module'] = self.__module__

        self._serialized = json_pack(self.config)
        self._uuid = uuid(self._serialized.encode())

        self.name = self.config['class']
        self.module = self.config['module']

        self._modified = {'a': None, 'u': None}

        # self.model here refers to classifiers, preprocessors and, possibly,
        # some representation of pipelines or the autoML itself.
        # Another possibility is to generalize modules to a new class Module()
        # that has self.model.
        self.unfit = True
        self.model = None

        # Each apply() uses a different training data, so this uuid is mutable
        self._train_data_uuid__mutable = None

        self.locked_by_others = False
        self.failed = False
        self.time_spent = None
        self.node = None
        self.failure = None
        self.show_warns = True
        self._param = None

    @classmethod
    def tree(cls, **kwargs):
        """
        Each tree represents a set of hyperparameter spaces and is a, possibly
        infinite, set of configurations.
        Parameters
        ----------
        data
            If given, 'data' limits the search interval of some hyperparameters

        Returns
        -------
            Tree representing all the possible hyperparameter spaces.
        """
        tree = cls.tree_impl(**kwargs)
        if 'config_spaces' in kwargs:
            del kwargs['config_spaces']
        hps = tree.hps.copy()

        hps['module'] = CatHP(choice, a=[cls.__module__])
        hps['class'] = CatHP(choice, a=[cls.__name__])

        # Freeze args passed via kwargs
        for k, v in kwargs.items():
            hps[k] = FixedHP(v)

        return tree.updated(hps=hps)

    def buildo(self, **config):
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
        # copia de config? ou seria inutil?
        # obj_copied.config.update(config)
        obj_copied.config = config

        # Create an encoded (uuid) unambiguous (sorted) version of config.
        obj_copied._serialized = 'building'
        obj_copied._uuid = uuid(obj_copied.serialized().encode())

        # TODO: which init vars should be restarted here?
        obj_copied.failure = None

        obj_copied.build_impl(**obj_copied.config)
        return obj_copied

    def apply(self, data=None):
        """Todo the doc string
        """
        # Checklist / get from storage -----------------------------------
        self.check_if_built()
        if data is None:
            self.msg(f"Applying {self.name} on None returns None.")
            return None  # If the Pipeline is done, that's ok.

        self._train_data_uuid__mutable = data.uuid()

        # TODO: CV() is too cheap to be recovered from storage,
        #  specially if it is a LOO.
        #  Maybe some components could inform whether they are cheap.
        output_data, started, ended = None, False, False
        if self._storage is not None:
            output_data, started, ended = \
                self._storage.get_result(self, 'a', data)

        if started:
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
        if not ended:
            if self._storage is not None:
                self._storage.lock(self, 'a', data)

            self.handle_warnings()
            if self.name != 'CV':
                self.msg('Applying ' + self.name + '...')
            start = self.clock()
            self.failure = None
            try:
                if self.max_time is None:
                    output_data = self.apply_impl(data)
                else:
                    with time_limit(self.max_time):
                        output_data = self.apply_impl(data)
            except Exception as e:
                print(e)
                self.failed = True
                self.failure = str(e)
                self.locked_by_others = False
                handle_exception(self, e)
            self.time_spent = self.clock() - start
            # self.msg('Component ' + self.name + ' applied.')
            self.dishandle_warnings()

            if self._storage is not None:
                self._storage.store_result(self, 'a', data, output_data)

        return output_data

    def use(self, data=None):
        """Todo the doc string
        """
        self.check_if_applied()

        # Checklist / get from storage -----------------------------------
        if data is None:
            self.msg(f"Using {self.name} on None returns None.")
            return None

        output_data, started, ended = None, False, False
        if self._storage is not None:
            output_data, started, ended = \
                self._storage.get_result(self, 'u', data)

        if started:
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
        if not ended:
            if self._storage is not None:
                self._storage.lock(self, 'u', data)

                # If the component was applied (probably simulated by storage),
                # but there is no model, we reapply it...
                if self.model is None:
                    print('It is possible that a previous apply() was '
                          'successfully stored, but its use() wasn\'t.'
                          'Or you are trying to use in new data.')
                    print(
                        'Trying to recover training data from storage to apply '
                        'just to induce a model usable by use()...\n'
                        f'comp: {self.sid()}  data: {data.sid()} ...')
                    train_uuid = self.train_data_uuid__mutable()
                    stored_train_data = \
                        self._storage.get_data_by_uuid(train_uuid)
                    self.model = self.apply_impl(stored_train_data)

            self.handle_warnings()
            if self.name != 'CV':
                print('Using ', self.name, '...')

            # TODO: put time limit and/or exception handling like in apply()?
            start = self.clock()
            output_data = self.use_impl(data)  # TODO:handl excps like in apply?
            self.time_spent = self.clock() - start

            # self.msg('Component ' + self.name + 'used.')
            self.dishandle_warnings()

            if self._storage is not None:
                self._storage.store_result(self, 'u', data, output_data)
        return output_data

    @classmethod
    def isdeterministic(cls):
        return False

    _ps = '''ps.: All Data transformation must be done via method updated() with 
        explicit keyworded args (e.g. X=X, y=...)!
        This is needed because modifies() will inspect the code and look for 
        the fields that can be modified by the component.'''

    @abstractmethod
    def apply_impl(self, data):
        f"""
                
        {self._ps} 
        """

    @abstractmethod
    def use_impl(self, data):
        f"""

        {self._ps}
        """

    # @abstractmethod
    # def explain(self, X):
    #     """Explain prediction/transformation for the given instances.
    #     """
    #     raise NotImplementedError("Should it return probability\
    #                                distributions, rules?")

    @classmethod
    @abstractmethod
    def tree_impl(cls, **kwargs):
        """Todo the doc string
        """
        pass

    @classmethod
    def check_data(cls, data):
        if data is None:
            raise Exception(cls.__name__ + ' needs a dataset to be able to \
                            estimate maximum values for some hyperparameters.')

    # @classmethod
    # def check_tree(cls, tree):
    #     try:
    #         node = tree.node
    #     except Exception as e:
    #         print(e)
    #         print()
    #         print(cls.__name__, ' <- problematic class')
    #         print()
    #         raise Exception('Problems with hyperparameter space')
    #
    #     try:
    #         for k in node:
    #             t = node[k][0]
    #             v = node[k][1]
    #             if t == 'c' or t == 'o':
    #                 if not isinstance(v, list):
    #                     raise Exception('Categorical and ordinal \
    #                                     hyperparameters need a list of \
    #                                     values: ' + str(k))
    #             else:
    #                 if len(v) != 2:
    #                     raise Exception('Real and integer hyperparameters need'
    #                                     ' a limit with two values: ' + str(k))
    #     except Exception as e:
    #         print(e)
    #         print()
    #         print(cls.__name__)
    #         print()
    #         raise Exception('Problems with hyperparameter space: ' + str(node))
    #
    #     for child in tree.children:
    #         cls.check_tree(child)

    def __str__(self, depth=''):
        return self.name + " " + str(self.config)

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
        return self._serialized

    def serializedo(self):
        """
        Calculate a representation of this built component.
        In the first call, remove reserved words from config
         (like 'mark', 'name').
        :return: 19-byte string
        """
        if self._serialized is None:
            raise ApplyWithoutBuild('build() should be called before '
                                    'serialized() <-' + self.name)

        # The first time serialized() is called,
        # it has to put needed (and remove unneeded) args from config.
        if self._serialized == 'building':
            # When the build is not created by a config coming from a HPTree,
            #  it can be lacking a name, so here we put it.
            if 'name' not in self.config:
                self.config['name'] = self.name

            # 'description' is needed because some components are not entirely
            # described by build() args.
            # self.config['description'] = self.describe()

            # Create an unambiguous (sorted) version of
            # config (build args+name+max_time) + init_vars (description).
            self._serialized = json_pack(self.config)

        return self._serialized

    @staticmethod
    def clock():
        t = os.times()
        # return t[4]  # Wall time
        return t[0] + t[1] + t[2] + t[3]

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
            raise UseWithoutApply(f'{self.name} should be applied after a '
                                  f'build!')

    def check_if_built(self):
        self.serialized()  # Call just to raise exception, if needed.

    def sid(self):
        """
        Short uuID
        First 5 chars of uuid for printing purposes.
        :return:
        """
        return self.uuid()[:5]

    @abstractmethod
    def modifies(self, op):
        pass

    def next(self):
        pass

    def param(self):
        if self._param is None:
            self._param = self.config.copy()
            del self._param['class']
            del self._param['module']

        return self._param
