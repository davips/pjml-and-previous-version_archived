from pymfe.mfe import MFE
from paje.base.component import Component
import numpy as np

from paje.base.data import Data


class SupMtFe(Component):
    def apply_impl(self, data):
        self.model.fit(*data.Xy)
        names, values = self.model.extract(suppress_warnings=True)
        print(values, 'datakkkkkkkkkkkkk')
        print([values])
        return Data(X=np.array([values]))

    def use_impl(self, data):
        self.model.fit(*data.Xy)
        names, values = self.model.extract(suppress_warnings=True)
        print(values, 'datakkkkkkkkkkkkk')
        print([values])
        return Data(X=np.array([values]))

    def build_impl(self):
        self.model = MFE()
