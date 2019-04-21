from sklearn.cluster import FeatureAgglomeration

from paje.base.component import Component
from paje.base.hps import HPTree
from paje.data.data import Data

# Data reduction by feature agglomeration
'''
Feature agglomeration

Example:
from paje.preprocessing.data_reduction.DRFtAg import DRFtAg
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

# create a DRFtAg instance
fa = DRFtAg(data)

# apply feature agglomeration to reduce n to 2 collumns
rd = fa.apply(2)
'''


class DRFtAg(Component):
    def __init__(self, data):
        self.x, self.y = data.xy()
        self.att_labels = data.columns

    def apply_impl(self, n_components, affinity='euclidean', linkage='ward'):
        if (linkage == 'ward'):
            affinity = 'euclidean'

        fa = FeatureAgglomeration(n_clusters=n_components, affinity='euclidean', linkage='ward')
        pc = fa.fit_transform(self.x)

        return Data(pc, self.y)

    def use_impl(self, data):
        pass

    @classmethod
    def hps_impl(cls, data=None):
        cls.check_data(data)
        return HPTree(
            dic={'n_components': ['z', list(range(1, data.n_attributes() + 1))],
                 'affinity': ['c', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']],
                 'linkage': ['c', ['ward', 'complete', 'average', 'single']]},
            children=[])
