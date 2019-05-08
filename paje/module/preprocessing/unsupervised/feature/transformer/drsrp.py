from sklearn.random_projection import SparseRandomProjection
from math import sqrt

# Data reduction by SRP
from paje.module.preprocessing.unsupervised.feature.transformer.reductor import Reductor

'''
This class is an sparse random projections implementation for data reduction.


Example:
from paje.preprocessing.data_reduction.DRSRP import DRSRP
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

# create a DRSRP instance
srp = DRSRP(data)

# apply SRP to reduce n to 2 collumns
rd = srp.apply(2)
'''


class DRSRP(Reductor):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)

    @classmethod
    def specific_dictionary(cls, data):
        return {
            # TODO: check if data.n_attributes() is correct here and in the line below
            # TODO: WTF is this sqrt?
            'density': ['o', [1 / sqrt(data.n_attributes()), 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
            'dense_output': ['c', [False, True]],
            'eps': ['o', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}


    def instantiate_model(self):
        self.model = SparseRandomProjection(**self.dict)


