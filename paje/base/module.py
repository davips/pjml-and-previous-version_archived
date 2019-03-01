# -*- coding: utf-8 -*-
"""Module the doc striing.
"""

from abc import ABC, abstractmethod


class Module(ABC):
    """ TODO doc string
    """

    @abstractmethod
    def fit(self, data_x, data_y=None, data_z=None):
        """ TODO doc string
        """

    @abstractmethod
    def use(self, data_x, data_y=None):
        """ TODO doc string
        """

    @abstractmethod
    def set_hs(self, hps):
        """ TODO doc string
        """

    @abstractmethod
    def get_hs(self):
        """ TODO doc string
        """
