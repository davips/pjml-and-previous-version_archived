from abc import abstractmethod
import numpy
from paje.base.component import Component
from paje.base.exceptions import handle_exception, UseWithoutApply
from paje.util.time import time_limit


class NonIterable(Component):

    _ps = '''ps.: All Data transformation must be done via method updated() with 
        explicit keyworded args (e.g. X=X, y=...)!
        This is needed because modifies() will inspect the code and look for 
        the fields that can be modified by the component.'''

    @abstractmethod
    def apply_impl(self, data):
        f"""

        {self._ps} 
        """

    @abstractmethod
    def use_impl(self, data):
        f"""

        {self._ps}
        """

    def apply(self, data=None):
        """Todo the doc string
        """
        if data is None:
            print(f"Applying {self.name} on None returns None.")
            return None  # If the Pipeline is done, that's ok.

        self._train_data_uuid__mutable = data.uuid()

        # TODO: CV() is too cheap to be recovered from storage,
        #  specially if it is a LOO.
        #  Maybe some components could inform whether they are cheap.
        output_data, started, ended = None, False, False
        if self._storage is not None:
            output_data, started, ended = \
                self._storage.get_result(self, 'a', data)

        if started:
            if self.failed:
                print(f"Won't apply on data {data.name}"
                         f"\nCurrent {self.name} already failed before.")
                return output_data

            if self.locked_by_others:
                print(f"Won't apply {self.name} on data "
                      f"{data.name}\n"
                      f"Currently probably working at node [{self.node}].")
                return output_data

        # Apply if still needed  ----------------------------------
        if not ended:
            if self._storage is not None:
                self._storage.lock(self, 'a', data)

            self.handle_warnings()
            if self.name != 'CV':
                print('Applying ' + self.name + '...')
            start = self.clock()
            self.failure = None
            try:
                if self.max_time is None:
                    output_data = self.apply_impl(data)
                else:
                    with time_limit(self.max_time):
                        output_data = self.apply_impl(data)
            except Exception as e:
                print(e)
                self.failed = True
                self.failure = str(e)
                self.locked_by_others = False
                handle_exception(self, e)
            self.time_spent = self.clock() - start
            # self.msg('Component ' + self.name + ' applied.')
            self.dishandle_warnings()

            if self._storage is not None:
                self._storage.store_result(self, 'a', data, output_data)

        return output_data

    def use(self, data=None):
        """Todo the doc string
        """
        self.check_if_applied()

        # Checklist / get from storage -----------------------------------
        if data is None:
            print(f"Using {self.name} on None returns None.")
            return None

        output_data, started, ended = None, False, False
        if self._storage is not None:
            output_data, started, ended = \
                self._storage.get_result(self, 'u', data)

        if started:
            if self.locked_by_others:
                print(f"Won't use {self.name} on data "
                         f"{data.name}\n"
                         f"Currently probably working at {self.node}.")
                return output_data

            if self.failed:
                print(f"Won't use on data {data.sid()}\n"
                         f"Current {self.name} already failed before.")
                return output_data

        # Use if still needed  ----------------------------------
        if not ended:
            if self._storage is not None:
                self._storage.lock(self, 'u', data)

                # If the component was applied (probably simulated by storage),
                # but there is no model, we reapply it...
                if self.model is None:
                    print('It is possible that a previous apply() was '
                          'successfully stored, but its use() wasn\'t.'
                          'Or you are trying to use in new data.')
                    print(
                        'Trying to recover training data from storage to apply '
                        'just to induce a model usable by use()...\n'
                        f'comp: {self.sid()}  data: {data.sid()} ...')
                    train_uuid = self.train_data_uuid__mutable()
                    stored_train_data = \
                        self._storage.get_data_by_uuid(train_uuid)
                    self.model = self.apply_impl(stored_train_data)

            self.handle_warnings()
            if self.name != 'CV':
                print('Using ', self.name, '...')

            # TODO: put time limit and/or exception handling like in apply()?
            start = self.clock()
            output_data = self.use_impl(data)  # TODO:handl excps like in apply?
            self.time_spent = self.clock() - start

            # self.msg('Component ' + self.name + 'used.')
            self.dishandle_warnings()

            if self._storage is not None:
                self._storage.store_result(self, 'u', data, output_data)
        return output_data

    def handle_warnings(self):
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        if not self.show_warns:
            numpy.warnings.filterwarnings('ignore')

    def dishandle_warnings(self):
        if not self.show_warns:
            numpy.warnings.filterwarnings('always')

    def check_if_applied(self):
        if self._train_data_uuid__mutable is None:
            raise UseWithoutApply(f'{self.name} should be applied after a '
                                  f'build!')

    def sid(self):
        """
        Short uuID
        First 5 chars of uuid for printing purposes.
        :return:
        """
        return self.uuid[:10]