import arff
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as ds
from paje.util.encoders import pack_data, uuid, uuid_enumerated_dic
from sklearn.utils import check_X_y


# TODO: convert in dataclass
class Data:

    def __init__(self, name, X=None, Y=None, Z=None, P=None,
                 U=None, V=None, W=None, Q=None,
                 columns=None, transformations=None):
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
        :param component: component that generated current Data
        :param transformations: History of transformations suffered by the data
        """

        # Init instance matrices and matrices_including_Nones to factory new
        # instances in the future.
        args = {k: v for k, v in locals().items() if v is not None
                and k != 'self' and k != 'name' and k != 'columns'
                and k != 'component'}
        if transformations is None:
            print('Warning: no component provided to Data, we will assume it '
                  'did\'t come from a transformation/prediction etc.')

        self.__dict__.update(args)
        matrices = args.copy()
        self._set('matrices', matrices)  # TODO: is copy really needed here?
        prediction = {k: v for k, v in matrices if k in ['Z', 'W', 'P', 'Q']}

        # Metadata
        n_classes = len(set(dematrixify(get_first_non_none([Y, V, Z, W]), [0])))
        n_instances = len([] and matrices[0])
        n_attributes = len(get_first_non_none([X, U], [[]])[0])

        self.__dict__.update({
            'n_classes': n_classes,
            'n_instances': n_instances,
            'n_attributes': n_attributes,
            'Xy': (X, dematrixify(Y)),
            'Uv': (U, dematrixify(V)),
            'prediction': prediction,
            'transformations': transformations or [],
            'matrices': matrices,
            # TODO: make columns effective, and save it to storage also
            'columns': None,
            'has_prediction_data': bool(prediction)
        })

        # Add vectorized shortcuts for matrices.
        vectors = ['y', 'z', 'v', 'w']
        self._set('vectors', vectors)
        for vec in vectors:
            self.__dict__[vec] = dematrixify(self.__dict__[vec.upper()])

        # Add lazy cache for dump and uuid
        self._set('_dump', None)
        self._set('_dump_prediction', None)
        self._set('_uuid', None)
        self._set('_dump_uuid', None)
        self._set('_name_uuid', None)
        self._set('_name', name)
        self._set('_fields', None)

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
        ps. Assume there was no transformations on this Data.
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
        return Data(name=arq, X=X, Y=Y, columns=df.columns,
                    transformations=[])

    @staticmethod
    def random(n_attributes, n_classes, n_instances):
        X, y = ds.make_classification(n_samples=n_instances,
                                      n_features=n_attributes,
                                      n_classes=n_classes,
                                      n_informative=int(
                                          np.sqrt(2 * n_classes)) + 1)
        return Data(X=X, Y=as_column_vector(y), transformations=[])

    @staticmethod
    def read_from_storage(name, storage, fields=None):
        """
        To just recover an original dataset you can pass fields='X,y'
        (case insensitive)
        or just None to recover as many fields as stored at the moment.
        :param name:
        :param fields: if None, get complete Data, including predictions if any
        :return:
        """
        return storage.get_data_by_name(name, fields)

    def store(self, storage):
        storage.store_data(self)

    def updated(self, component, **kwargs):
        """
        Return a new Data updated by given values.
        :param component: to put into transformations list for history purposes
        (it can be a list of transformations also for internal use in Data).
        :param kwargs:
        :return:
        """
        new_kwargs = self.matrices.copy()
        new_kwargs.update(kwargs)
        for l in self.vectors:
            if l in new_kwargs:
                L = l.upper()
                new_kwargs[L] = as_column_vector(new_kwargs.pop(l))
        if 'name' not in new_kwargs:
            new_kwargs['name'] = self.name()

        if 'transformations' not in new_kwargs:
            new_kwargs['transformations'] = \
                self.transformations + [component.serialized()]
        else:
            print('Warning, giving transformations from outside update().')
        return Data(**new_kwargs)

    def merged(self, data):
        """
        Get more matrices (or new values) from another Data.
        The longest history will be kept.
        They should have the same name and be different by at most one
        transformation.
        Otherwise, an exception will be raised.
        :param data:
        :return:
        """
        dica = uuid_enumerated_dic(data.transformations)
        dicb = uuid_enumerated_dic(self.transformations)
        uuids = set(dica.keys()).symmetric_difference(set(dicb.keys()))
        if len(uuids) > 1:
            raise Exception(f'Merging {self.name()}: excess of '
                            f'transformations in one of the Data instances',
                            data.transformations, self.transformations)
        transformations= data.transformations if len(dica) > len(dicb)\
            else self.transformations
        if self.name() != data.name():
            raise Exception(f'Merging {self.name()} with {data.name()}')
        return self.updated(transformations=transformations, **data.matrices)

    def select(self, fields):
        """
        Return a subset of the dictionary of kwargs.
        ps.: Automatically convert vectorized shortcuts to matrices.
        :param fields:
        :return:
        """
        fields_lst = fields.split(',')
        matrixnames = [(x.upper() if x in self.vectors else x)
                       for x in fields_lst]

        # Raise exception if any requested matrix is None.
        if any([m not in self.matrices for m in matrixnames]):
            raise Exception('Requested None matrix', fields, self.matrices.keys)

        return {k: v for k, v in self.matrices.items() if k in matrixnames}

    def reduced_to(self, fields):
        return Data(name=self.name(), transformations=self.transformations,
                    **self.select(fields))

    def __setattr__(self, attr, value):
        raise MutabilityException(
            'Cannot set attributes on Data! (%s %r)'
            % (self.__class__.__name__, attr))

    def _set(self, name, value):
        object.__setattr__(self, name, value)

    def dump(self):
        if self._dump is None:
            # This if is needed to avoid useless redumping of the same data.
            if self.is_prediction_data:
                self._set('_dump', self.dump_prediction_only())
            else:
                self._set('_dump', pack_data(self.matrices))
        return self._dump

    def dump_prediction_only(self):
        if self._dump_prediction is None:
            self._set('_dump_prediction', pack_data(self.prediction))
        return self._dump_prediction

    def uuid(self):
        if self._uuid is None:
            # The scenario when a dataset with the same name and fields
            # has more than a row in the storage is when different
            # models provide different dataset predictions/transformations.
            # This is solved by adding the history_uuid of transformations
            # into the data.UUID.
            # Opting by a dual key composed by both name-fields-history and
            # UUID, we have faster UUID calculations than calculating UUID
            # on the matrices dump; and also enforce human readable integrity
            # (name and fields and history are more readable than Data.UUID).
            uuid_ = uuid(self.name_uuid() + self.fields() + self.history_uuid())
            self._set('_uuid', uuid_)
        return self._uuid

    def name_uuid(self):
        if self._name_uuid is None:
            self._set('_name_uuid', uuid(self.name()))
        return self._name_uuid

    def dump_uuid(self):
        if self._dump_uuid is None:
            self._set('_dump_uuid', uuid(self.dump()))
        return self._dump_uuid

    def name(self):
        if self._name is None:
            self._set('_name', f'unnamed[{self.uuid()}]')
        return self._name

    def fields(self):
        if self._fields is None:
            sorted = list(self.matrices.keys())
            if 'columns' in sorted:
                sorted.remove('columns')
            sorted.sort()
            self._set('_fields', ','.join(sorted))
        return self._fields

    def __str__(self):
        txt = []
        [txt.append(f'{k}: {str(v)}') for k, v in self.matrices.items()]
        return '\n'.join(txt) + self.name()

    def split(self, train_size = 0.25, random_state=1):
        X, y = self.Xy
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
        train_size = 0.25  # TODO: set it as parameter
        name = f'{self.name()}_seed{random_state}_split{train_size}_fold'
        trainset = Data(name=name, X=X_train).updated(y=y_train)
        testset = Data(name=name, X=X_test).updated(y=y_test)
        #TODO: usar component CV()

        return trainset, testset

    def shapes(self):
        """
        Return the shape of all matrices.
        :return:
        """
        return [v.shape() for v in self.matrices.values()]


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
