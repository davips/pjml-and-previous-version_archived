# -*- coding: utf-8 -*-
""" Composer module

This module implements and describes the 'Pipeline' composer.  This composer
sequentially operates on 'Elements' component or 'Composers'.

For more information about the Composer concept see [1].

.. _paje_arch Paje Architecture:
    TODO: put the link here
"""

import copy

from paje.automl.composer.composer import Composer
from paje.base.hps import ConfigSpace


class Pipeline(Composer):
    def build_impl(self, **config):
        """
        The only parameter is config with the dic of each component.
        :param config
        :return:
        """
        configs = [{} for _ in self.components]  # Default value

        if 'configs' in self.config:
            configs = self.config['configs']
        self.components = self.components.copy()
        zipped = zip(range(0, len(self.components)), configs)
        for idx, compo_config in zipped:
            # TODO: setar showwarns?
            newconfig = compo_config.copy()
            newconfig['random_state'] = self.random_state
            print(55555555, self.components[idx], newconfig)
            self.components[idx] = self.components[idx].build(**newconfig)

    @classmethod
    def tree_impl(cls, config_spaces):
        bottom = ConfigSpace.bottom()

        current = bottom
        print(current)
        for config_space in reversed(config_spaces):
            current = config_space.updated(children=[current])
            print(current)
        top = ConfigSpace.top(name='Pipeline', children=[current])

        return ConfigSpace(start=top, end=bottom)

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in self.components]
        return self.name + " {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n' \
               + depth + "}"
