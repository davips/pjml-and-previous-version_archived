import zlib
import copy
import json
import hashlib

import arff
import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.utils import check_X_y

from paje.result.sqlite import SQLite
from paje.result.storage import uuid


class Data(object):
    def __init__(self, data_x, data_y, columns=None):
        self.is_classification = False
        self.is_regression = False
        self.is_clusterization = False

        self.is_supervised = False
        self.is_unsupervised = False

        self.already_serialized = None

        check_X_y(data_x, data_y)
        # TODO
        # check if exits dtype indefined == object

        if data_x is not None:
            if data_y is not None:
                self.is_supervised = True
                if issubclass(data_y.dtype.type, np.floating):
                    self.is_regression = True
                else:
                    self.is_classification = True
            else:
                self.is_clusterization = True

        self.p = None  # Predicted probabilities
        self.v = None  # x of unlabeled set
        self.w = None  # y of unlabeled set
        self.data_x = data_x
        self.data_y = data_y
        self.z = None
        self.columns = columns

    @staticmethod
    def read_arff(file, target):
        data = arff.load(open(file, 'r'), encode_nominal=True)

        columns = data["attributes"]
        df = pd.DataFrame(data['data'],
                          columns=[attr[0] for attr in data['attributes']])

        data_y = df.pop(target).values
        data_x = df.values.astype('float')

        return Data(data_x, data_y, columns)

    @staticmethod
    def random(n_attributes, n_classes, n_instances):
        x, y = ds.make_classification(n_samples=n_instances, n_features=n_attributes, n_classes=n_classes, n_informative=int(np.sqrt(2 * n_classes)) + 1)
        return Data(x, y)

    def read_csv(file, target):
        raise NotImplementedError("Method read_csv should be implement!")

    def xy(self):
        return self.data_x, self.data_y

    def pvwxyz(self):
        return self.p, self.v, self.w, self.data_x, self.data_y, self.z

    def copy(self):
        return copy.deepcopy(self)

    def n_instances(self):
        return len(self.data_x)

    def n_attributes(self):
        return len(self.data_x[0])

    def n_classes(self):
        # Unfortunately, it is impossible to memoize this calculation because Data() is promiscuously mutable and accepts external changes to data_x and _y from everyone.
        return len(set(self.data_y))

    def serialized(self):
        # TODO: should Data be immutable, we could memoize everything
        # if self.already_serialized is None:    ## already_serialized cannot be part of self, because of copy() nao aceitando memoryview() (but memoryview is not used anymore)
        already_serialized = SQLite.pack(self.pvwxyz())
        # self.already_serialized = zlib.compress(json.dumps(self.pvwxyz(), sort_keys=True).encode())
        return already_serialized

    def __hash__(self):
        return uuid(self.serialized())
