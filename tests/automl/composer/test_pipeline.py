from paje.base.data import Data
from paje.automl.composer.pipeline import Pipeline
import numpy as np
import pytest

@pytest.fixture
def get_composer(get_elements, simple_data):
    aaa, bbb, ccc, ddd = get_elements
    data = simple_data

    compr = Pipeline(components=[aaa, bbb, ccc, ddd])
    args_sets = {
        'name':compr.name,
        'dics':[
            {'name': aaa.name, 'oper': '+'},
            {'name': bbb.name, 'oper': '.'},
            {'name': ccc.name, 'oper': '+'},
            {'name': ddd.name, 'oper': '*'},]
    }
    print(compr.tree().tree_to_dict())
    mycompr = compr.build(**args_sets)
    data_apply = mycompr.apply(data)
    data_use = mycompr.use(data)

    return (aaa, bbb, ccc, ddd, mycompr, data_apply, data_use, data)


def test_apply_use(get_composer):
    aaa, _, _, _, _, data_apply, data_use, data = get_composer

    # sequence made in apply
    X = data.X
    X_aaa = X + X
    X_bbb = np.dot(X_aaa, X_aaa)
    X_ccc = X_bbb + X_bbb
    X_ddd = X_ccc * X_ccc
    assert np.allclose(X_ddd, data_apply.X)

    # sequence made in use
    X += X_aaa
    X += X_bbb
    X += X_ccc
    X += X_ddd
    assert np.allclose(X, data_use.X)
