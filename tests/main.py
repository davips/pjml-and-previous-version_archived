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


hp = HPSpace(name="FS")
hp.add_axis(hp, "k", 'd', 1, 1, np.random.randint)
#
# dt = hp.add_branch(hp, "decision tree")
# hp.add_axis(dt, "alg", 'd', 1, 1, 'dt')
# hp.add_axis(dt, "pr1", 'd', 1, 1, 'dt')
# pr2_1 = hp.add_branch(dt, "PR2_1")
# hp.add_axis(pr2_1, "pr2", 'd', 1, 1, 'dt')
# hp.add_axis(pr2_1, "pr4", 'd', 1, 1, 'dt')
# pr2_2 = hp.add_branch(dt, "PR2_2")
# hp.add_axis(pr2_2, "pr2", 'd', 1, 1, 'dt')
# hp.add_axis(pr2_2, "pr3", 'd', 1, 1, 'dt')
#
# svm = hp.add_branch(hp, "support vector machine")
# hp.add_axis(svm, "alg", 'd', 1, 1, 'rf')
#
# hp.print(data=True)
#
#
def objective(*argv, **kwargs):
    print("*argv --> {0} \n **kwargs --> {1}".format(argv, kwargs))
    alg = kwargs.get('alg')

    print("alg={0}\npr1={1}\npr2={2}\npr3={3}\npr4={4}\n\n".format(alg, pr1, pr2, pr3, pr4))
    return 1

sr = RandomSearch(hp)
sr.fmin(objective)
