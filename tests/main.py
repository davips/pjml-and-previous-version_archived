
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# adding the project root to import 'paje' modules
PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.split(PATH)[0]
sys.path.append(MODULE_PATH)

from paje.preprocessing.feature_selection.SelectKBest import SelectKBest
from paje.opt.random_search import RandomSearch
from paje.opt.hp_space import HPSpace


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


def my_func():
    return np.random.randint(1, 10)

def making_space():
    hp = HPSpace(name="root")
    hp.add_axis(hp, "x1", 'c', 0, 5, ['5', '10', '15', '20', '25'])

    b1 = hp.new_branch(hp, "b1")
    hp.add_axis(b1, "x2", 'r', 0, 10, np.random.ranf)
    hp.add_axis(b1, "x3", 'z', -2, 10, np.random.ranf)

    b2 = hp.new_branch(hp, "b2")
    hp.add_axis(b2, "x4", 'f', None, None, my_func)

    hp.print(data=True)

    return hp


def objective(*argv, **kwargs):
    print("*argv --> {0} \n **kwargs --> {1}".format(argv, kwargs))

    aux = 10000
    x3 = kwargs.get('x3')
    x1 = kwargs.get('x1')
    x4 = kwargs.get('x4')

    if x3 != None:
        x2 = kwargs.get('x2')
        aux = int(x1) + x2 + x3 + 1
    elif x4 != None:
        aux = int(x1) * x4
    else:
        print("Some error occur")


    return aux


np.random.seed(0)
sr = RandomSearch(making_space())
print()
result = sr.fmin(objective)
print("\nBest fmin = {0}\nConf = {1}\n".format(result[0], result[1]))

