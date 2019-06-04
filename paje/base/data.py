import arff
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as ds
from sklearn.utils import check_X_y

from paje.result.storage import uuid, pack_data


# TODO: convert in dataclass
class Data:

    def __init__(self, name, X=None, Y=None, Z=None, P=None,
                 U=None, V=None, W=None, Q=None,
                 columns=None):
        """
            Immutable lazy data for all machine learning scenarios
             we could imagine.
             Y, Z, V and W can be accessed as vectors y, z, v and w
             if there is only one output.
             Otherwise, they are multitarget matrices.

        :param name:
        :param X:
        :param Y:
        :param Z: Predictions
        :param P: Predicted probabilities
        :param U: X of unlabeled set
        :param V: Y of unlabeled set
        :param W: Predictions for unlabeled set
        :param Q: Predicted probabilities for unlabeled set
        :param columns:
        """
        # Init instance vars and dic to factory new instances in the future.
        args = {k: v for k, v in locals().items() if
                k != 'self' and k != 'name' and k != 'columns'}
        self.__dict__.update(args)
        dic = args.copy()
        self._set('_dic', dic)

        alldata = X, Y, Z, U, V, W, P, Q

        # Metadata
        n_classes = len(set(dematrixify(get_first_non_none([Y, V, Z, W]), [0])))
        n_instances = len(get_first_non_none(alldata, []))
        n_attributes = len(get_first_non_none([X, U], [[]])[0])
        used_vars = {k: v for k, v in self._dic.items() if v is not None}

        self.__dict__.update({
            'n_classes': n_classes,
            'n_instances': n_instances,
            'n_attributes': n_attributes,
            'Xy': (X, dematrixify(Y)),
            'Uv': (U, dematrixify(V)),
            'all': alldata,
            'used_vars': used_vars,
            'columns': None  # TODO: make columns effective, and save it to
            #     storage also
        })

        # Add vectorized shortcuts for matrices.
        vectors = ['y', 'z', 'v', 'w']
        self._set('vectors', vectors)
        for vec in vectors:
            self.__dict__[vec] = dematrixify(self.__dict__[vec.upper()])

        # Add lazy cache for dump and uuid
        self._set('_dump', None)
        self._set('_uuid', None)
        self._set('_name', name)
        self._set('_fields_str', None)

        # Check list
        if not isinstance(name, str):
            raise Exception('Wrong parameter passed as name', name)
        if X is not None and Y is not None:
            try:
                check_X_y(X, self.__dict__['y'])
            except Exception as e:
                print('X', X)
                print('Y', Y)
                print()
                print('X', X.shape)
                print('Y', Y.shape)
                raise e

        # TODO
        # check if exits dtype indefined == object
        # check dimensions of all matrices and vectors

        # TODO:   Could we check this through Noneness of z,u,v,w?
        # self._set('is_classification', False)
        # self._set('is_regression', False)
        # self._set('is_clusterization', False)
        #
        # self.is_supervised = False
        # self.is_unsupervised = False
        # if X is not None:
        #     if y is not None:
        #         self.is_supervised = True
        #         if issubclass(y.dtype.type, np.floating):
        #             self.is_regression = True
        #         else:
        #             self.is_classification = True
        #     else:
        #         self.is_clusterization = True

    @staticmethod
    def read_arff(file, target):
        data = arff.load(open(file, 'r'), encode_nominal=True)
        df = pd.DataFrame(data['data'],
                          columns=[attr[0] for attr in data['attributes']])
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_csv(file, target):
        df = pd.read_csv(file)
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_data_frame(df, file, target):
        Y = as_column_vector(df.pop(target).values.astype('float'))
        X = df.values.astype('float')
        arq = file.split('/')[-1]
        return Data(name='file_' + arq, X=X, Y=Y, columns=df.columns)

    @staticmethod
    def random(n_attributes, n_classes, n_instances):
        X, y = ds.make_classification(n_samples=n_instances,
                                      n_features=n_attributes,
                                      n_classes=n_classes,
                                      n_informative=int(
                                          np.sqrt(2 * n_classes)) + 1)
        return Data(name=None, X=X, Y=as_column_vector(y))


    @staticmethod
    def read_from_storage(name, storage):
        return storage.get_data_by_name(name)

    def updated(self, **kwargs):
        dic = self._dic.copy()
        dic.update(kwargs)
        for l in self.vectors:
            if l in dic:
                L = l.upper()
                dic[L] = as_column_vector(dic.pop(l))
        if 'name' not in dic:
            dic['name'] = self.name()
        return Data(**dic)

    def merged(self, data):
        """
        Same as updated, but not replacing any field by None.
        :param data:
        :return:
        """
        return self.updated(**data.used_vars)

    def shrink(self):
        """
        Remove None fields
        :return:
        """
        return Data(name=self.name(), **self.used_vars)

    def select(self, fields):
        """
        Return a subset of the dictionary of kwargs.
        ps.:Convert vectorized shortcuts to matrices.
        :param fields:
        :return:
        """
        fields = [(x.upper() if x in self.vectors else x) for x in fields]
        return {k: v for k, v in self._dic.items() if k in fields}

    def reduced_to(self, fields):
        return Data(name=self.name(), **self.select(fields))

    def __setattr__(self, attr, value):
        raise MutabilityException(
            'Cannot set attributes on Data! (%s %r)'
            % (self.__class__.__name__, attr))

    def _set(self, name, value):
        object.__setattr__(self, name, value)

    def dump(self):
        if self._dump is None:
            self._set('_dump', pack_data(self._dic))
        return self._dump

    def uuid(self):
        if self._uuid is None:
            self._set('_uuid', uuid(self.dump()))
        return self._uuid

    def name(self):
        if self._name is None:
            self._set('_name', f'unnamed[{self.uuid()}]')
        return self._name

    def fields_str(self):
        if self._fields_str is None:
            sorted = list(self.used_vars.keys())
            if 'columns' in sorted:
                sorted.remove('columns')
            sorted.sort()
            self._set('_fields_str', ','.join(sorted))
        return self._fields_str

    def __str__(self):
        txt = []
        [txt.append(f'{k}: {str(v)}') for k, v in self.used_vars.items()]
        return '\n'.join(txt) + self.name()

    def split(self, random_state=1):
        X, y = self.Xy
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
        train_size = 0.25  # TODO: set it as parameter
        name = f'{self.name()}_seed{random_state}_split{train_size}_fold'
        trainset = Data(name=name + '0', X=X_train).updated(y=y_train)
        testset = Data(name=name + '1', X=X_test).updated(y=y_test)
        return trainset, testset


class MutabilityException(Exception):
    pass


def as_vector(mat):
    return mat.reshape(len(mat))


def as_column_vector(vec):
    return vec.reshape(len(vec), 1)


def dematrixify(m, default=None):
    return default if m is None else as_vector(m)


def get_first_non_none(l, default=None):
    """
    Consider the first non None list in the args for extracting metadata.
    :param l:
    :return:
    """
    filtered = list(filter(None.__ne__, l))
    return default if filtered == [] else filtered[0]
