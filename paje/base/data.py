from logging import warning

import arff
import numpy as np
import pandas as pd
import sklearn.datasets as ds

from paje.base.chain import Chain
from paje.ml.element.preprocessing.supervised.instance.sampler.cv \
    import CV
from paje.util.encoders import pack_data, uuid, json_unpack, zlibext_pack


class Data:
    """ Data
    """
    _vectors = {i: i.upper() for i in ['y', 'z', 'v', 'w', 'e', 'f']}
    _scalars = {'r': 'E', 's': 'F', 't': 'k'}
    _unshortcut = {v: k for k, v in _vectors.items()}
    _unshortcut.update({v: k for k, v in _scalars.items()})
    _unshortcut.update({
        'X': 'X',
        'U': 'U',
        'l': 'l',
        'm': 'm',
        'S': 'S'
    })
    print(9999999999999999, _unshortcut)

    def __init__(self, name,
                 X, Y=None, Z=None, P=None,
                 U=None, V=None, W=None, Q=None,
                 E=None, F=None,
                 l=None, m=None,
                 k=None,
                 S=None,
                 columns=None, history=None):
        """
            Immutable lazy data for all machine learning scenarios
             we could imagine.
             Matrices Y, Z, V and W can be accessed as vectors y, z, v and w
             if there is only one output.
             Otherwise, they are multitarget matrices.

             Vector e (or f) summarizes e.g. some operation over y,z (or v,w)
             k is a vector indicating the time step of each row.
             Vector k has the time steps of each instance, t has a single time
             step for the entire chunk (all rows in Data). t == k[0]

             For instance, when the vectors e,f,k have only one value,
             which is the most common scenario, they can be accessed as values
             r,s,t. r == e[0] and s == f[0]

             Vector of metafeatures 'm' can have any length, which is
             dependent on the MF() component used and explicited in history.

        :param name:

        :param X: attribute values
        :param Y: labels
        :param Z: Predictions
        :param P: Predicted probabilities
        :param e: Results, summarized in some way

        :param U: X of unlabeled set
        :param V: Y of unlabeled set
        :param W: Predictions for unlabeled set
        :param Q: Predicted probabilities for unlabeled set
        :param f: Results for unlabeled, summarized in some way

        :param k: Time steps for the current chunk (this Data), one per row

        :param l: supervised metafeatures characterizing this Data's X and y
        :param m: unsupervised metafeatures characterizing this Data's X+U

        :param columns: attribute names
        :param history: History of transformations suffered by the data

        #:param modified: list of fields modified in the last trasnformation

        :param task: intended use for this data: classification, clustering,...
        :param n_classes: this could be calculated,
            but it is too time consuming to do it every transformation.
        """
        # ALERT: all single-letter args will be considered matrices/vectors!
        self._fields = {k: v for k, v in locals().items() if len(k) == 1}
        self._fields['S'] = Chain() if S is None else S
        self.__dict__.update(self._fields)

        self.name = f'unnamed[{self.uuid()}]' if name is None else name
        self._history = Chain() if history is None else history

        if Y is not None:
            self.n_classes = np.unique(Y).shape[0]
        self.n_instances = X.shape[0]
        self.n_attributes = X.shape[1]

        # Add vectorized shortcuts for matrices.
        for k, v in self._vectors.items():
            self.__dict__[k] = self._dematrixify(self.__dict__[v])

        # Add single-valued shortcuts for vectors.
        for k, v in self._scalars.items():
            self.__dict__[k] = self._devectorize(self.__dict__[v])

        # Add lazy cache for dump and uuid
        for var in self._fields:
            self.__dict__['_dump' + var] = None
            self.__dict__['_uuid' + var] = None

        self._uuid = None
        self._history_uuid = None
        self._name_uuid = None
        self.Xy = X, self.__dict__['y']
        self.columns = columns

    def field_dump(self, field_name):
        """
        Dump of the matrix/vector associated to the given field.
        :param field_name:
        :return: binary compressed dump
        """
        if field_name not in self._fields:
            raise Exception(f'Field {field_name} not available in this Data!')

        key = '_dump' + field_name
        if self.__dict__[key] is None:
            self.__dict__[key] = pack_data(self._fields[field_name])
        return self.__dict__[key]

    def field_uuid(self, field_name):
        """
        UUID of the matrix/vector associated to the given field.
        :param field_name: Case sensitive.
        :return: UUID
        """
        key = '_uuid' + field_name
        if self.__dict__[key] is None:
            uuid_ = uuid(self.field_dump(field_name))
            self.__dict__[key] = uuid_
        return self.__dict__[key]

    # remover ?
    def uuids_dumps(self):
        """
        :return: pair uuid-dump of each matrix/vector.
        """
        return {self.field_uuid(k): self.field_dump(k) for k in self._fields}

    # remover ?
    def uuids_fields(self):
        """
        :return: pair uuid-field of each matrix/vector.
        """
        return {self.field_uuid(k): k for k in self._fields}

    @staticmethod
    def read_arff(file, target=None):
        """
        Create Data from ARFF file.
        See read_data_frame().
        :param file:
        :param target:
        :return:
        """
        data = arff.load(open(file, 'r'), encode_nominal=True)
        df = pd.DataFrame(data['data'],
                          columns=[attr[0] for attr in data['attributes']])
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_csv(file, target=None):
        """
        Create Data from CSV file.
        See read_data_frame().
        :param file:
        :param target:
        :return:
        """
        df = pd.read_csv(file)  # 1169_airlines explodes here with RAM < 6GiB
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_data_frame(df, file, target=None):
        """
        ps. Assume X,y classification task.
        Andd that there was no transformations (history) on this Data.
        :param df:
        :param file: name of the dataset (if a path, name will be extracted)
        :param target:
        :return:
        """
        name = file.split('/')[-1]
        Y = target and Data._as_column_vector(
            df.pop(target).values.astype('float'))
        X = df.values.astype('float')  # Do not call this before setting Y!
        return Data(name=name, X=X, Y=Y, columns=list(df.columns))

    @staticmethod
    def random_classification(n_attributes, n_classes, n_instances):
        """
        ps. Assume X,y classification task.
        :param n_attributes:
        :param n_classes:
        :param n_instances:
        :return:
        """
        X, y = ds.make_classification(n_samples=n_instances,
                                      n_features=n_attributes,
                                      n_classes=n_classes,
                                      n_informative=int(
                                          np.sqrt(2 * n_classes)) + 1)
        return Data(X=X, Y=Data._as_column_vector(y))

    @staticmethod
    def read_from_storage(name, storage, fields=None):
        """
        To just recover an original dataset you can pass fields='X,Y'
        (case sensitive)
        or just None to recover as many fields as stored at the moment.
        :param name:
        :param storage:
        :param fields: if None, get complete Data, including predictions if any
        :return:
        """
        return storage.get_data_by_name(name, fields)

    def store(self, storage):
        storage.store_data(self)

    def updated(self, component, **kwargs):
        """ Return a new Data updated by given values.
        :param component: for history purposes
        :param kwargs:
        :return:
        """
        new_args = self._fields.copy()
        for field, value in kwargs.items():
            if field in self._vectors:
                new_args[self._vectors[field]] = self._as_column_vector(value)
            elif field in self._scalars:
                new_args[self._scalars[field]] = np.array(value, ndmin=2)

        new_args['name'] = kwargs['name'] if 'name' in kwargs else self.name

        new_args['columns'] = kwargs['columns'] \
            if 'columns' in kwargs else self.columns

        if 'history' in kwargs:
            new_args['history'] = kwargs['history']
        else:
            new_args['history'] = Chain(component.config, self._history)

        if 'S' in kwargs:
            new_args['S'] = kwargs['S']

        return Data(**new_args)

    def _get(self, name):
        return object.__getattribute__(self, self._unshortcut[name])

    def sid(self):
        """
        Short uuID
        First 10 chars of uuid for printing purposes.
        Max of 1 collision each 1048576 combinations.
        :return:
        """
        return self.uuid()[:10]

    def uuid(self):
        if self._uuid is None:
            # The scenario when a dataset with the same name and fields
            # has more than a row in the storage is when different
            # models provide different dataset predictions/transformations.
            # This is solved by adding the history_uuid of transformations
            # into the data.UUID.
            self._uuid = uuid((self.name_uuid() + self.history_uuid()).encode())

        return self._uuid

    def name_uuid(self):
        if self._name_uuid is None:
            self._name_uuid = uuid(self.name.encode())
        return self._name_uuid

    def history_uuid(self):
        if self._history_uuid is None:
            self._history_uuid = uuid(zlibext_pack(str(self._history)))
        return self._history_uuid

    def __str__(self):
        txt = []
        [txt.append(f'{k}: {str(v)}') for k, v in self._fields.items()]
        return '\n'.join(txt) + "name" + self.name + "\n" + \
               "history=" + str(self.history()) + "\n"

    def split(self, test_size=0.25, random_state=1):
        cv = CV(config={'random_state': random_state, 'split': 'holdout',
                        'test_size': test_size, 'steps': 1, 'iteration': 0})
        return cv.apply(self), cv.use(self)

    def history(self):
        return str(self._history)

    @staticmethod
    def _as_vector(mat):
        s = mat.shape[0]
        return mat.reshape(s)

    @staticmethod
    def _as_column_vector(vec):
        return vec.reshape(len(vec), 1)

    @staticmethod
    def _dematrixify(m, default=None):
        return default if m is None else Data._as_vector(m)

    @staticmethod
    def _devectorize(v, default=None):
        return default if v is None else v[0]

    # Todo: Jogar fora?
    def field_names(self) -> str:
        if self._fields is None:
            sortd = list(self._fields.keys())
            sortd.sort()
            self._set('_fields', ','.join(sortd))
        return self._fields
