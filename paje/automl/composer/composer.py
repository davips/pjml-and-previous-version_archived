# -*- coding: utf-8 -*-
""" Composer module

This module is a generic module to build a 'Composer'. The idea of the Composer
is to make operations over 'Components'. For example, the Pipeline is a
Composer that applies Components sequentially. On the other hand, the Switch
Composer applies only one Component at a time.

For more information about the Composer concept see [1].

.. _paje_arch Paje Architecture:
    TODO: put the link here
"""
from abc import ABC, abstractmethod

from paje.base.component import Component
from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace
from paje.util.misc import flatten, get_class


class Composer(Component, ABC):
    """Core Composer class.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.components = []
        for subconfig in self.config['configs']:
            aux = get_class(subconfig['module'], subconfig['class'])
            if aux.isdeterministic():
                subconfig['random_state'] = self.config['random_state']
            self.components.append(aux(subconfig))
        self.model = 42

    def apply_impl(self, data):
        """ This function will be called by Component in the the 'apply()' step.

        Attributes
        ----------
        data: :obj:`Data`
            The `Data` object that represent a dataset used for training fase.
        """

        for component in self.components:
            data = component.apply(data)
            if component.failed:
                raise Exception('Applying subcomponent failed! ', component)
        return data

    def use_impl(self, data):
        """ This function will be called by Component in the the 'use()' step.

        Attributes
        ----------
        data: :obj:`Data`
            The `Data` object that represent a dataset used for testing phase.
        """

        for component in self.components:
            data = component.use(data)
            if component.failed:
                raise Exception('Using subcomponent failed! ', component)
        return data

    def modifies(self, op):
        if op not in ['a', 'u']:
            raise Exception('Wrong op:', op)

        if self._modified[op] is None:
            self._modified[op] = list(set(
                flatten([compo.modifies(op) for compo in self.components])
            ))
        return self._modified[op]

    @staticmethod
    @abstractmethod
    def sampling_function(config_spaces):
        pass

    @classmethod
    def tree_impl(cls, config_spaces):
        hps = [CatHP('configs', cls.sampling_function,
                     config_spaces=config_spaces),
               ]
        return ConfigSpace(name='Switch', hps=hps)

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in self.components]
        return self.name + " {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n' \
               + depth + "}"
