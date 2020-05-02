import json
from abc import abstractmethod

import numpy

from pjdata.aux.decorator import classproperty
from pjml.tool.abc.mixin.timers import Timers


class ExceptionHandler:
    """Handle transformer exceptions and enable/disable numpy warnings.

        E.g. Mahalanobis distance in KNN needs to supress warnings due to NaN
        in linear algebra calculations. MLP is also verbose due to
        nonconvergence issues among other problems.
    """

    @classproperty
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    def _handle_warnings():
        numpy.warnings.filterwarnings('ignore')

    @staticmethod
    def _dishandle_warnings():
        numpy.warnings.filterwarnings('always')

    msgs = ['All features are either constant or ignored.',  # CB
            'be between 0 and min(n_samples, n_features)',  # DR*
            'excess of max_free_parameters:',  # MLP
            'Timed out!',
            'Mahalanobis for too big data',
            'MemoryError',
            'On entry to DLASCL parameter number',  # Mahala knn
            'excess of neighbors!',  # KNN
            'subtransformer failed',  # nested failure
            'specified nu is infeasible',  # SVM
            'excess of neurons',
            'max_depth must be greater than zero'
            ]
    transformer = None

    def _handle_exception(self, e, exit_on_error):
        """Pipeline failure is different from python error."""
        if isinstance(self.name, str):
            # TODO: remove the need for this IF, if it still exists
            print(f'At {self.name},\nTrying to handle:\n[{str(e)}]')
        else:
            print(f'At {self.name()},\nTrying to handle:\n[{str(e)}]')
        if not any([str(e).__contains__(msg) for msg in self.msgs]):

            # HINTS
            if str(e).__contains__('cannot perform reduce with flexible type') \
                    or str(e).__contains__(
                'could not convert string to float'):
                from pjml.tool.data.processing.feature.binarize import Binarize
                print(f'HINT: your pipeline may be missing a '
                      f'{Binarize.name} component')

            # end of handling
            # print('TODO: is exit_on_error implemented? exit_on_error=',
            #       exit_on_error)
            # if exit_on_error:
            #     traceback.print_exc()
            #     print('Exiting...')
            #     exit(0)
            # else:
            raise e
        else:
            print(' just a known pipeline failure.'
                  'Will be put onto Data object.')

    def _check_nodata(self, data, transformer):
        from pjdata.specialdata import NoData
        if data is NoData and not transformer.nodata_handler:
            raise Exception(f'NoData is not accepted by {self.name}!')

    def _check_history(self, datain, dataout, transformations):
        """Check consistency between resulting Data object and
        _transformations() implementation provided by the current
        component."""
        # TODO: global(?) option to disable history checking (takes << 200us)
        # st = Timers._clock()
        from pjdata.specialdata import NoData
        if dataout is NoData:  # or dataout.isfrozen or dataout.allfrozen:
            return dataout

        # Predict output UUID.
        expected = datain.uuid
        for trf in transformations:
            expected *= trf.uuid

        # # Traverse actual history.
        # actual = datain.uuid
        # print('actul', actual)
        # print(33333333333333333333333333333333333333)
        # print(datain.history)
        # print(4444444444444444444444444444444444444)
        # print(dataout.history)
        # print(5555555555555555555555555555)
        # for trf in dataout.history[len(datain.history):]:
        #     print(trf)
        #     actual += trf.uuid
        #     print(actual)
        # print('-==========================')

        # Check if expected uuid is the same as the one from original data.
        if expected != dataout.uuid:
            recent = dataout.history[len(datain.history):]
            print('\nActual history::::::::::::::: data:', dataout.uuid)
            for trf in recent:
                print(f'{trf}')

            print(f'\nExpected history:::::::::::::::: data:', expected)
            for trf in transformations:
                print(f'{trf}')

            print(f'\nTransformed Data object history does not '
                  'match expected transformation list.\n'
                  'Please override self._transformations() '
                  f'method for {self.name} or extend a proper parent class '
                  f'like \'Invisible\'.\n')

            print('in:', type(datain), datain)
            print('out:', type(dataout), dataout)

            raise BadComponent(f'Inconsistent Data object history!')

        # print((Timers._clock() - st)*1000, 'ms')
        return dataout


class MissingModel(Exception):
    pass


class BadComponent(Exception):
    pass
