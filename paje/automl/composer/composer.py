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

from paje.base.component import Component
from paje.util.misc import flatten


class Composer(Component):
    """Core Composer class.
    """

    def __init__(self, components=None, **kwargs):
        """The class provides easy access for metafeature extraction from
        datasets.

        ps. It should transform data only through its internal components,
        never directly (to avoid inconsistencies, e.g. the need to
        override the method modifies()).

        Attributes
        ----------
        components : :obj:`list`
            A list with Components.
        **kwargs
            The parent class attributes.

        """

        # TODO: An empty Pipeline/composer may return perfect predictions.
        super().__init__(**kwargs)
        if components is None:
            components = []

        # before build()
        self.components = components

        # Updated after build()
        self.random_state = 0
        self.mytree = None

        # TODO: better model here?
        self.model = 42

    def describe(self):
        """The describing function called to uniquely identify the object.
        """

        if self._describe is None:
            self._describe = {
                'module': self.module,
                'name': self.name,
                'sub_components': [comp.describe() for comp in self.components]
            }
        return self._describe

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
