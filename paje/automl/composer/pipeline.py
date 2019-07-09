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
            self.components[idx] = self.components[idx].build(**newconfig)

    # def set_leaf(self, tree, f):
    #     if len(tree.children) > 0:
    #         for child in tree.children:
    #             self.set_leaf(child, f)
    #     else:
    #         if not (tree.name and tree.name.startswith('End')
    #                 and tree.tmp_uuid == self.tmp_uuid):
    #             tree.children.append(f())
    #
    def tree_impl(self, config_spaces):
        new_config_space = ConfigSpace('Pipeline')

        aux = new_config_space.start()
        for config_space in config_spaces:
            aux.add_child(config_space.start())
            aux = config_space.end()
        new_config_space.finish([aux])

        return new_config_space


    # def tree_impl_(self):
    #     if self.mytree is None:
    #         trees = []
    #         for i in range(0, len(self.components)):
    #             # TODO: Why were we using deepcopy here?
    #             tree = copy.copy(self.components[i]).tree()
    #             trees.append(tree)
    #
    #         self.set_leaf(trees[len(trees) - 1], lambda:
    #         HPTree({}, [], name='End' + self.name, tmp_uuid=self.tmp_uuid))
    #         for i in reversed(range(1, len(trees))):
    #             self.set_leaf(trees[i - 1], lambda: trees[i])
    #
    #         self.mytree = HPTree({}, [trees[0]], name=self.name)
    #
    #     return self.mytree

    def __str__(self, depth=''):
        newdepth = depth + '    '
        strs = [component.__str__(newdepth) for component in self.components]
        return self.name + " {\n" + \
               newdepth + ("\n" + newdepth).join(str(x) for x in strs) + '\n' \
               + depth + "}"
