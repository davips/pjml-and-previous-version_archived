import arff
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as ds
from paje.module.experimenting.cv import CV
from paje.util.encoders import pack_data, uuid, uuid_enumerated_dic, json_unpack
from sklearn.utils import check_X_y


def sub(a, b):
    return [item for item in a if item not in b]


# TODO: convert in dataclass
class Data:

    def __init__(self, name, X=None, Y=None, Z=None, P=None, e=None,
                 U=None, V=None, W=None, Q=None, f=None,
                 k=None,
                 columns=None, history=None, task=None):
        # ALERT: all single-letter args will be considered matrices/vectors!
        """
            Immutable lazy data for all machine learning scenarios
             we could imagine.
             Matrices Y, Z, V and W can be accessed as vectors y, z, v and w
             if there is only one output.
             Otherwise, they are multitarget matrices.

             Vector e (or f) summarizes e.g. some operation over y,z (or v,w)
             k is a vector indicating the time step of each row.
             However, when the vectors e,f,k have only one value,
             which is the most common scenario, they can be accessed as values
             r,s,t.

        :param name:
        :param X:
        :param Y:
        :param Z: Predictions
        :param P: Predicted probabilities
        :param e: Results, summarized in some way
        :param U: X of unlabeled set
        :param V: Y of unlabeled set
        :param W: Predictions for unlabeled set
        :param Q: Predicted probabilities for unlabeled set
        :param f: Results for unlabeled, summarized in some way
        :param k: Time steps for the current chunk (this Data)
        :param columns:
        :param component: component that generated current Data
        :param history: History of transformations suffered by the data
        """
        if history is None:
            print('No history provided to Data')
            history = []

        # Init matrices/vectors to factory new Data instances in the future.
        matvecs = {k: v for k, v in locals().items()  # Only used ones.
                   if v is not None and len(k) == 1}
        all_mats_vecs = {k: v for k, v in locals().items() if len(k) == 1}

        self.__dict__.update(all_mats_vecs)
        prediction = {k: v for k, v in matvecs.items()
                      if k in ['Z', 'W', 'P', 'Q']}

        # Metadata
        n_classes = len(set(dematrixify(get_first_non_none([Y, V, Z, W]), [0])))
        n_instances = len([] or get_first_non_none(matvecs.values()))
        n_attributes = len(get_first_non_none([X, U], [[]])[0])

        self.__dict__.update({
            '_n_classes': n_classes,
            '_n_instances': n_instances,
            '_n_attributes': n_attributes,
            # As a convenient notation for Data, matrices, vectors and
            # single values are the only members available directly as
            # variables, not functions.
            'Xy': (X, dematrixify(Y)),
            'Uv': (U, dematrixify(V)),
            '_prediction': prediction,
            '_history': history or [],
            '_matvecs': matvecs,
            '_columns': columns,
            # '_has_prediction_data': bool(prediction),
            # '_is_prediction_data': bool(set(prediction.keys()) -
            #                             set(matvecs.keys()))
        })

        # Add vectorized shortcuts for matrices.
        vectors = ['y', 'z', 'v', 'w']
        self._set('_vectors', vectors)
        for vec in vectors:
            self.__dict__[vec] = dematrixify(self.__dict__[vec.upper()])

        # Add single-valued shortcuts for vectors.
        self._set('_val2vec', {'r': 'e', 's': 'f', 't': 'k'})
        self._set('_single_values', self._val2vec.keys())  # TODO: needed?
        for k, v in self._val2vec.items():
            self.__dict__[k] = devectorize(v)

        # Add lazy cache for dump and uuid
        self._set('_dump', None)
        self._set('_dump_prediction', None)
        self._set('_uuid', None)
        self._set('_history_uuid', None)
        self._set('_dump_uuid', None)
        self._set('_name_uuid', None)
        self._set('_name', name)
        self._set('_fields', None)

        # Check list
        if not isinstance(name, str):
            raise Exception('Wrong parameter passed as name=', name)
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

        # TODO: how to properly identify the task?
        #  Does a dataset really has an inherent task related to it?
        #  Should it be defined by the user?
        self._set('_is_classification', None)
        self._set('_is_regression', None)
        self._set('?is_clusterization', None)
        self._set('is_supervised', None)
        self._set('is_unsupervised', None)
        self._set('is_multilabel', None)
        self._set('is_ranking_prediction', None)

    @staticmethod
    def read_arff(file, target, storage=None):
        data = arff.load(open(file, 'r'), encode_nominal=True)
        df = pd.DataFrame(data['data'],
                          columns=[attr[0] for attr in data['attributes']])
        return Data.read_data_frame(df, file, target, storage)

    @staticmethod
    def read_csv(file, target, storage=None):
        """
        Create Data from CSV file.
        :param file:
        :param target:
        :param storage: Where to look for previously stored data if possible,
        to avoid useless waits loading from file and recalculating UUID.
        :return:
        """
        df = pd.read_csv(file)
        return Data.read_data_frame(df, file, target, storage)

    @staticmethod
    def read_data_frame(df, file, target, storage=None):
        """
        ps. Assume there was no transformations (history) on this Data.
        :param df:
        :param file:
        :param target:
        :param storage:
        :return:
        """
        arq = file.split('/')[-1]
        data = storage and \
               Data.read_from_storage(name=arq, storage=storage, fields='X,y')
        if data is not None:
            return data
        X = df.values.astype('float')
        Y = as_column_vector(df.pop(target).values.astype('float'))
        return Data(name=arq, X=X, Y=Y, columns=list(df.columns), history=[])

    @staticmethod
    def random(n_attributes, n_classes, n_instances):
        X, y = ds.make_classification(n_samples=n_instances,
                                      n_features=n_attributes,
                                      n_classes=n_classes,
                                      n_informative=int(
                                          np.sqrt(2 * n_classes)) + 1)
        return Data(X=X, Y=as_column_vector(y), history=[])

    @staticmethod
    def read_from_storage(name, storage, fields=None):
        """
        To just recover an original dataset you can pass fields='X,y'
        (case insensitive)
        or just None to recover as many fields as stored at the moment.
        :param name:
        :param storage:
        :param fields: if None, get complete Data, including predictions if any
        :return:
        """
        return storage.get_data_by_name(name, fields)

    def store(self, storage):
        storage.store_data(self)

    def updated(self, component_or_list, **kwargs):
        """ Return a new Data updated by given values.
        :param component_or_list: to put into transformations list for history purposes
        (it can be a list also for internal use in Data).
        :param kwargs:
        :return:
        """
        new_args = self.matvecs().copy()
        new_args.update(kwargs)
        for l in self.vectors():
            if l in new_args:
                L = l.upper()
                new_args[L] = as_column_vector(new_args.pop(l))
        for k, v in self.val2vec():
            if k in new_args:
                new_args[v] = as_vector(new_args.pop(k))
        if 'name' not in new_args:
            new_args['name'] = self.name()

        if 'history' not in new_args:
            if isinstance(component_or_list, list):
                new_args['history'] = self.history() + component_or_list
            else:
                new_args['history'] = self.history() + [
                    json_unpack(component_or_list.serialized())
                ]
        else:
            print('Warning: giving \'history\' from outside update().')
        return Data(columns=self.columns(), **new_args)

    def merged(self, new_data):
        """
        Get more matrices/vectors (or new values) from another Data.
        The longest history will be kept.
        They should have the same name and be different by at least one
        transformation.
        Otherwise, an exception will be raised.
        new_data has precedence over self.
        :param new_data:
        :return:
        """
        # checking...
        if self.name() != new_data.name():
            raise Exception(f'Merging {self.name()} with {new_data.name()}')

        if new_data.history()[:len(self.history())] != self.history():
            print(new_data.history(), '\n', self.history())
            raise Exception('Incompatible transformations, self.history '
                            'should be the start of new_data.history')

        history = new_data.history()
        if len(history) == len(self.history()):
            history += ['Merge']

        dic = self.matvecs().copy()
        dic.update(new_data.matvecs())

        return Data(name=self.name(), history=history, columns=self.columns(),
                    **dic)

    def select(self, fields):
        """
        Return a subset of the dictionary of kwargs.
        ps.: Automatically convert vectorized/single-valued shortcuts to
        matrices/vectors.
        ps 2: ignore inexistent fields
        ps 3: raise exception in none fields
        :param fields: 'all' means 'don't touch anything'
        'except:z,w' means keep all except z,w
        :return:
        """
        fields.replace(':', ',')
        if fields == 'all':
            fields_lst = self.fields()
        else:
            fields_lst = fields.split(',')
            if fields_lst and fields_lst[0] == 'except':
                fields_lst = sub(self.fields(), fields_lst[1:])

        namesmats = [(x.upper() if x in self.vectors() else x)
                     for x in fields_lst]
        namesvecs = [vec for val, vec in self._val2vec.items()
                     if val in fields_lst]
        names = namesmats + namesvecs

        # Raise exception if any requested matrix is None.
        if any([mv not in self.matvecs() for mv in names]):
            raise Exception('Requested None or inexistent matrix/vector/value',
                            fields, self.matvecs().keys)

        return {k: v for k, v in self.matvecs().items() if k in names}

    def shrink_to(self, fields):
        return Data(name=self.name(), history=self.history(),
                    columns=self.columns(), **self.select(fields))

    def __setattr__(self, attr, value):
        raise MutabilityException(
            'Cannot set attributes on Data! (%s %r)'
            % (self.__class__.__name__, attr))

    def _set(self, name, value):
        object.__setattr__(self, name, value)

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
            uuid_ = uuid((self.name_uuid() + self.history_uuid()).encode())
            self._set('_uuid', uuid_)
        return self._uuid

    def name_uuid(self):
        if self._name_uuid is None:
            self._set('_name_uuid', uuid(self.name().encode()))
        return self._name_uuid

    def history_uuid(self):
        if self._history_uuid is None:
            self._set('_history_uuid', uuid(','.join(self.history()).encode()))
        return self._history_uuid

    def name(self):
        if self._name is None:
            self._set('_name', f'unnamed[{self.uuid()}]')
        return self._name

    def fields(self):
        if self._fields is None:
            sorted = list(self.matvecs().keys())
            sorted.sort()
            self._set('_fields', ','.join(sorted))
        return self._fields

    def __str__(self):
        txt = []
        [txt.append(f'{k}: {str(v)}') for k, v in self.matvecs().items()]
        return '\n'.join(txt) + self.name()

    def split(self, test_size=0.25, random_state=1):
        cv = CV().build(random_state=random_state, split='holdout',
                        test_size=test_size, steps=1, testing_fold=0)
        return cv.apply(self), cv.use(self)

    def shapes(self):
        """
        Return the shape of all matrices/vectors.
        :return:
        """
        return {k: v.shape for k, v in self.matvecs().items()}

    # Redefinig all class member as functions just for uniformity and 
    # autocompleteness. And to show protection against changes. 
    def n_classes(self):
        return self._n_classes

    def n_instances(self):
        return self._n_instances

    def n_attributes(self):
        return self._n_attributes

    def prediction(self):
        return self._prediction

    def history(self):
        return self._history

    def matvecs(self):
        return self._matvecs

    def columns(self):
        return self._columns

    def has_prediction_data(self):
        return self._has_prediction_data

    def vectors(self):
        return self._vectors


class MutabilityException(Exception):
    pass


def as_vector(mat):
    return mat.reshape(len(mat))


def as_column_vector(vec):
    return vec.reshape(len(vec), 1)


def dematrixify(m, default=None):
    return default if m is None else as_vector(m)


def devectorize(v, default=None):
    return default if v is None else v[0]


def get_first_non_none(l, default=None):
    """
    Consider the first non None list in the args for extracting metadata.
    :param l:
    :return:
    """
    filtered = list(filter(None.__ne__, l))
    return default if filtered == [] else filtered[0]
