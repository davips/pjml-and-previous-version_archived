from paje.preprocessing.feature_selection.SelectKBest import SelectKBest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

y = data1.pop("target")
y_disc = pd.qcut(y, 3, labels=["A", "B", "C"])
X = data1

for m in SelectKBest.get_methods():
    skb = SelectKBest(method=m, k=2)
    skb.fit(X, y_disc.cat.codes)
    X_new = skb.transform(X)
    print(skb.idx)
    print(X_new.columns)


