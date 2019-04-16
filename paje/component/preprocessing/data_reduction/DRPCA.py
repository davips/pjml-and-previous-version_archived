from sklearn.decomposition import PCA

from paje.component.component import Component
from paje.base.hps import HPTree
from paje.data.data import Data

# Data reduction by PCA
'''
This class is a PCA (principal component analysis) implementation for data reduction.

- given matrix A with dimension m (instances) x n (features) and d value, which is the new amount of features. PCA aims to reduce a matrix Amxn to Amxd

- In this method version, the z-score technique was not implemented


- PCA is applied in following steps:

    1 - feature standardization (PCA is sensible to the measure scale): A = StandardScaler().fit_transform(A)

    2 - Measure the average for each line of the matrix A: u = A.mean(0)

    3 - Measuring covariance: C = (1 / (m - 1)) * (A - u).T @ (A - u)

    4 - Get eigenvalues and eigenvectors: eig_vals, eig_vecs = numpy.linalg.eig(C)

    5 - Get eigenvector subset: W = eig_vecs[:, 0 : d]

    6 - PCA result: Y = A @ W # Y = A W


Example:
from paje.preprocessing.data_reduction.DRPCA import DRPCA
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

# create a DRPCA instance
pca = DRPCA(data)

# apply pca to reduce n to 2 collumns
rd = pca.apply(2)
'''


class DRPCA(Component):
    def __init__(self, **kwargs):
        # TODO: if StandardScaler is obligatory for PCA use, we need to integrate this kind of requirement into the Pajé architecture
        # # standardize features: PCA is sensible to the measure scale
        # self.x = StandardScaler().fit_transform(self.x)

        self.pca = PCA(**kwargs)

    def apply_impl(self, data):
        self.pca.fit_transform(data.data_x)

        return Data(pc, self.y)

    def use_impl(self, data):
        pass

    @staticmethod
    def hps_impl(data):
        return HPTree(
            dic={'n_components': ['z', list(range(1, data.n_attributes() + 1))]},
            children=[])
