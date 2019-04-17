from sklearn.random_projection import GaussianRandomProjection

from paje.component import Component
from paje.base.hps import HPTree
from paje.data.data import Data

# Data reduction by GRP
'''
This class is a Gaussian random projections implementation for data reduction.


Example:
from paje.preprocessing.data_reduction.DRGRP import DRSRP
from paje.data.data import Data
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# create a Data type instance
x = df.loc[:, features].values
y = df.loc[:,['target']].values
data = Data(x, y)

# create a DRGRP instance
grp = DRGRP(data)

# apply GRP to reduce n to 2 collumns
rd = grp.apply(2)
'''
class DRGRP(Component):
    def __init__(self, data):
        self.x, self.y = data.xy()
        self.att_labels = data.columns


    def apply_impl(self, n_components, eps = 0.1):
        rp = GaussianRandomProjection(n_components = n_components, eps = eps, random_state = 420)
        pc = rp.fit_transform(self.x)

        return Data(pc, self.y)


    def use_impl(self, data):
        pass

    @classmethod
    def hps_impl(cls, data=None):
        cls.check_data(data)
        return HPTree(
            dic={'n_components': ['z', list(range(1, data.n_attributes() + 1))],
                 'eps': ['r', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]},
            children=[])
