# from abc import abstractmethod, ABC
# from functools import lru_cache
# from typing import Union
#
# from pjdata.aux.decorator import classproperty
# from pjdata.aux.serialization import serialize, materialize
# from pjdata.data import Data
# from pjdata.mixin.identifyable import Identifyable
# from pjdata.mixin.printable import Printable
# from pjdata.step.transformation import Transformation
# from pjml.config.description.cs.configlist import ConfigList
# from pjml.tool.abc.mixin.exceptionhandler import BadComponent, ExceptionHandler
# from pjml.tool.abc.mixin.timers import Timers
# from pjml.tool.model.model import Model
# from pjml.tool.model.specialmodel import EarlyEndedModel, FailedModel
#
#
# class Transformer(Printable, Identifyable, ExceptionHandler, Timers, ABC):
#     """Parent of all processors, learners, evaluators, data controlers, ...
#
#     Contributors:
#
#     Each component (alias for Transformer child classes) implementation should
#     decide by itself if it requires the 'apply' step before the 'use' step.
#     self.model should be set at the time of calling use().
#
#     All components should implement:
#         _apply_impl()
#         _use_impl()
#         _cs_impl()
#
#     They also should call super.__init__(), assigning values to:
#         config¹
#         algorithm²
#         isdeterministic*
#     and at _apply_impl(), or before (at init), assign a value to:
#         self.model³
#
#     *: Deterministic components should override this class member with True.
#     ¹: algorithm parameters
#     ²: processor/learner/evaluator to apply()
#     ³: induced/fitted/describing model to use()
#     """
#
#     def __init__(self, config,
#                  deterministic=False, nodata_handler=False, max_time=3600):
#         jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
#         Printable.__init__(self, jsonable)
#
#         self.config = config
#         self.deterministic = deterministic
#         from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
#         self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler
#
#         self._exit_on_error = True
#
#         self.cs = self.cs1  # Shortcut to ease retrieving a CS from a
#         # Transformer object without having to check that it is not a
#         # component (Transformer class).
#
#         self.max_time = max_time  # TODO: who/when to define maxtime?
#
#     @abstractmethod
#     def _use_impl(self, data, **kwargs):
#         """Each component should implement its core 'apply' functionality."""
#
#     @abstractmethod
#     def apply(self, data, exit_on_error=False):
#         """Training step (usually).
#
#          Fit/remove-noise-from/evaluate/... Data.
#
#          Implementers: If your component requires apply() before use(),
#          it should extend mixin EnforceApply and _apply_impl() should
#          return (None, use_impl()) when provided data is None.
#
#          Parameters
#          ----------
#          data
#              'None' means 'pipeline ended before this transformation'.
#              'NoData' means 'pipeline alive, hoping to generate Data in
#              the next transformer'.
#          exit_on_error
#              Exit imediatly instead of just marking a failure inside Data
#              object.
#
#          Returns
#          -------
#          Transformed data, normally.
#          None, when data is a None
#              (probably meaning the pipeline finished before this transformer).
#          Same data, but annotated with a failure.
#
#          Exception
#          ---------
#          BadComponent
#              Data resulting history should be consistent with
#              _transformations() implementation.
#          """
#
#     @abstractmethod
#     def _apply_impl(self, data):
#         """Each component should implement its core 'apply' functionality."""
#
#     @classmethod
#     @abstractmethod
#     def _cs_impl(cls):
#         """Each component should implement its own 'cs'. The parent class
#         takes care of 'name' and 'path' arguments of ConfigSpace"""
#
#     # TODO: Is unbounded lrucache a source of memory leak?
#     @lru_cache()
#     def transformations(self, step, clean=True):
#         """Expected transformation described as a list of Transformation
#         objects.
#
#         Child classes should override this method to perform non-atomic or
#         non-trivial transformations.
#         A missing implementation will be detected during apply/use."""
#         if step in 'au':
#             return [Transformation(self, step)]
#         else:
#             raise BadComponent('Wrong current step:', step)
#
#     @classproperty
#     @lru_cache()
#     def cs(cls):
#         """Config Space of this component, when called as class method.
#         If called on an transformer (object/instance method), will convert
#         the object to a config space with a single transformer.
#
#         Each Config Space is a tree, where each path represents a parameter
#         space of the learning/processing/evaluating algorithm of this component.
#         It is a possibly infinite set of configurations.
#
#         Returns
#         -------
#             Tree representing all the possible parameter spaces.
#         """
#         cs_ = cls._cs_impl()
#         return cs_.identified(name=cls.__name__, path=cls.__module__)
#
#     @property
#     @lru_cache()
#     def cs1(self=None):
#         """Convert transformer into a config space with a single transformer
#         inside it."""
#         return ConfigList(self)
#
#     def updated(self, **kwargs):
#         """Clone this transformer, optionally replacing given params.
#
#         Returns
#         -------
#         A ready to use transformer.
#         """
#         config = self.config
#         if kwargs:
#             config = config.copy()
#             config.update(kwargs)
#         return materialize(self.name, self.path, config)
#
#     @property
#     @lru_cache()
#     def serialized(self):
#         return serialize(self)
#
#     @staticmethod
#     def _to_config(locals_):
#         """Convert a dict coming from locals() to config."""
#         config = locals_.copy()
#         del config['self']
#         del config['__class__']
#         return config
#
#     def _uuid_impl00(self):
#         return self.serialized
#
#     @classproperty
#     @lru_cache()
#     def name(cls):
#         return cls.__name__
#
#     @property
#     @lru_cache()
#     def longname(self):
#         return self.name
#
#     @classproperty
#     @lru_cache()
#     def path(cls):
#         return cls.__module__
#
#     @property
#     @lru_cache()
#     def wrapped(self):
#         """Same as unwrap(), but with the external container Wrap."""
#         return None
#
#     @property
#     @lru_cache()
#     def unwrap(self):
#         """Subpipeline inside the first Wrap().
#
#         Hopefully there is only one Wrap in the pipeline.
#         This method performs a depth-first search.
#
#         Example:
#         pipe = Pipeline(
#             File(name='iris.arff'),
#             Wrap(Std(), SVMC()),
#             Metric(function='accuracy')
#         )
#         pipe.unwrap  # -> Chain(Std(), SVMC())
#         """
#         return self.wrapped.transformer
#
#
# class UTransformer(Printable, Identifyable, ExceptionHandler, Timers, ABC):
#     """Parent of all processors, learners, evaluators, data controlers, ...
#
#     Contributors:
#
#     Each component (alias for Transformer child classes) implementation should
#     decide by itself if it requires the 'apply' step before the 'use' step.
#     self.model should be set at the time of calling use().
#
#     All components should implement:
#         _apply_impl()
#         _use_impl()
#         _cs_impl()
#
#     They also should call super.__init__(), assigning values to:
#         config¹
#         algorithm²
#         isdeterministic*
#     and at _apply_impl(), or before (at init), assign a value to:
#         self.model³
#
#     *: Deterministic components should override this class member with True.
#     ¹: algorithm parameters
#     ²: processor/learner/evaluator to apply()
#     ³: induced/fitted/describing model to use()
#     """
#
#     def __init__(self, config,
#                  deterministic=False, nodata_handler=False, max_time=3600):
#         jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
#         Printable.__init__(self, jsonable)
#
#         self.config = config
#         self.deterministic = deterministic
#         from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
#         self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler
#
#         self._exit_on_error = True
#
#         self.cs = self.cs1  # Shortcut to ease retrieving a CS from a
#         # Transformer object without having to check that it is not a
#         # component (Transformer class).
#
#         self.max_time = max_time  # TODO: who/when to define maxtime?
#
#     @abstractmethod
#     def _use_impl(self, data, **kwargs):
#         """Each component should implement its core 'apply' functionality."""
#
#     @abstractmethod
#     def apply(self, data, exit_on_error=False):
#         """Training step (usually).
#
#          Fit/remove-noise-from/evaluate/... Data.
#
#          Implementers: If your component requires apply() before use(),
#          it should extend mixin EnforceApply and _apply_impl() should
#          return (None, use_impl()) when provided data is None.
#
#          Parameters
#          ----------
#          data
#              'None' means 'pipeline ended before this transformation'.
#              'NoData' means 'pipeline alive, hoping to generate Data in
#              the next transformer'.
#          exit_on_error
#              Exit imediatly instead of just marking a failure inside Data
#              object.
#
#          Returns
#          -------
#          Transformed data, normally.
#          None, when data is a None
#              (probably meaning the pipeline finished before this transformer).
#          Same data, but annotated with a failure.
#
#          Exception
#          ---------
#          BadComponent
#              Data resulting history should be consistent with
#              _transformations() implementation.
#          """
#
#     @abstractmethod
#     def _apply_impl(self, data):
#         """Each component should implement its core 'apply' functionality."""
#
#     @classmethod
#     @abstractmethod
#     def _cs_impl(cls):
#         """Each component should implement its own 'cs'. The parent class
#         takes care of 'name' and 'path' arguments of ConfigSpace"""
#
#     # TODO: Is unbounded lrucache a source of memory leak?
#     @lru_cache()
#     def transformations(self, step, clean=True):
#         """Expected transformation described as a list of Transformation
#         objects.
#
#         Child classes should override this method to perform non-atomic or
#         non-trivial transformations.
#         A missing implementation will be detected during apply/use."""
#         if step in 'au':
#             return [Transformation(self, step)]
#         else:
#             raise BadComponent('Wrong current step:', step)
#
#     @classproperty
#     @lru_cache()
#     def cs(cls):
#         """Config Space of this component, when called as class method.
#         If called on an transformer (object/instance method), will convert
#         the object to a config space with a single transformer.
#
#         Each Config Space is a tree, where each path represents a parameter
#         space of the learning/processing/evaluating algorithm of this component.
#         It is a possibly infinite set of configurations.
#
#         Returns
#         -------
#             Tree representing all the possible parameter spaces.
#         """
#         cs_ = cls._cs_impl()
#         return cs_.identified(name=cls.__name__, path=cls.__module__)
#
#     @property
#     @lru_cache()
#     def cs1(self=None):
#         """Convert transformer into a config space with a single transformer
#         inside it."""
#         return ConfigList(self)
#
#     def updated(self, **kwargs):
#         """Clone this transformer, optionally replacing given params.
#
#         Returns
#         -------
#         A ready to use transformer.
#         """
#         config = self.config
#         if kwargs:
#             config = config.copy()
#             config.update(kwargs)
#         return materialize(self.name, self.path, config)
#
#     @property
#     @lru_cache()
#     def serialized(self):
#         return serialize(self)
#
#     @staticmethod
#     def _to_config(locals_):
#         """Convert a dict coming from locals() to config."""
#         config = locals_.copy()
#         del config['self']
#         del config['__class__']
#         return config
#
#     def _uuid_impl00(self):
#         return self.serialized
#
#     @classproperty
#     @lru_cache()
#     def name(cls):
#         return cls.__name__
#
#     @property
#     @lru_cache()
#     def longname(self):
#         return self.name
#
#     @classproperty
#     @lru_cache()
#     def path(cls):
#         return cls.__module__
#
#     @property
#     @lru_cache()
#     def wrapped(self):
#         """Same as unwrap(), but with the external container Wrap."""
#         return None
#
#     @property
#     @lru_cache()
#     def unwrap(self):
#         """Subpipeline inside the first Wrap().
#
#         Hopefully there is only one Wrap in the pipeline.
#         This method performs a depth-first search.
#
#         Example:
#         pipe = Pipeline(
#             File(name='iris.arff'),
#             Wrap(Std(), SVMC()),
#             Metric(function='accuracy')
#         )
#         pipe.unwrap  # -> Chain(Std(), SVMC())
#         """
#         return self.wrapped.transformer


from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Union

from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.data import Data
from pjdata.mixin.identifyable import Identifyable
from pjdata.mixin.printable import Printable
from pjdata.step.transformation import Transformation
from pjml.config.description.cs.configlist import ConfigList
from pjml.tool.abc.mixin.exceptionhandler import BadComponent, ExceptionHandler
from pjml.tool.abc.mixin.timers import Timers
from pjml.tool.model.model import Model
from pjml.tool.model.specialmodel import FailedModel, EarlyEndedModel


class UTransformer(Printable, Identifyable, ExceptionHandler, Timers, ABC):
    """Parent of all processors, learners, evaluators, data controlers, ...

    Contributors:

    Each component (alias for Transformer child classes) implementation should
    decide by itself if it requires the 'apply' step before the 'use' step.
    self.model should be set at the time of calling use().

    All components should implement:
        _apply_impl()
        _use_impl()
        _cs_impl()

    They also should call super.__init__(), assigning values to:
        config¹
        algorithm²
        isdeterministic*
    and at _apply_impl(), or before (at init), assign a value to:
        self.model³

    *: Deterministic components should override this class member with True.
    ¹: algorithm parameters
    ²: processor/learner/evaluator to apply()
    ³: induced/fitted/describing model to use()
    """

    def __init__(self, config,
                 deterministic=False, nodata_handler=False, max_time=3600):
        jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
        Printable.__init__(self, jsonable)

        self.config = config
        self.deterministic = deterministic
        from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
        self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler

        self._exit_on_error = True

        self.cs = self.cs1  # Shortcut to ease retrieving a CS from a
        # Transformer object without having to check that it is not a
        # component (Transformer class).

        self.max_time = max_time  # TODO: who/when to define maxtime?

    @abstractmethod
    def _use_impl(self, data):
        """Each component should implement its core 'apply' functionality."""

    @abstractmethod
    def apply(self, data, exit_on_error=False):
        """Training step (usually).

         Fit/remove-noise-from/evaluate/... Data.

         Implementers: If your component requires apply() before use(),
         it should extend mixin EnforceApply and _apply_impl() should
         return (None, use_impl()) when provided data is None.

         Parameters
         ----------
         data
             'None' means 'pipeline ended before this transformation'.
             'NoData' means 'pipeline alive, hoping to generate Data in
             the next transformer'.
         exit_on_error
             Exit imediatly instead of just marking a failure inside Data
             object.

         Returns
         -------
         Transformed data, normally.
         None, when data is a None
             (probably meaning the pipeline finished before this transformer).
         Same data, but annotated with a failure.

         Exception
         ---------
         BadComponent
             Data resulting history should be consistent with
             _transformations() implementation.
         """

    @abstractmethod
    def _apply_impl(self, data):
        """Each component should implement its core 'apply' functionality."""

    @classmethod
    @abstractmethod
    def _cs_impl(cls):
        """Each component should implement its own 'cs'. The parent class
        takes care of 'name' and 'path' arguments of ConfigSpace"""

    # TODO: Is unbounded lrucache a source of memory leak?
    @lru_cache()
    def transformations(self, step, clean=True):
        """Expected transformation described as a list of Transformation
        objects.

        Child classes should override this method to perform non-atomic or
        non-trivial transformations.
        A missing implementation will be detected during apply/use."""
        if step in 'au':
            return [Transformation(self, step)]
        else:
            raise BadComponent('Wrong current step:', step)

    @classproperty
    @lru_cache()
    def cs(cls):
        """Config Space of this component, when called as class method.
        If called on an transformer (object/instance method), will convert
        the object to a config space with a single transformer.

        Each Config Space is a tree, where each path represents a parameter
        space of the learning/processing/evaluating algorithm of this component.
        It is a possibly infinite set of configurations.

        Returns
        -------
            Tree representing all the possible parameter spaces.
        """
        cs_ = cls._cs_impl()
        return cs_.identified(name=cls.__name__, path=cls.__module__)

    @property
    @lru_cache()
    def cs1(self=None):
        """Convert transformer into a config space with a single transformer
        inside it."""
        return ConfigList(self)

    def updated(self, **kwargs):
        """Clone this transformer, optionally replacing given params.

        Returns
        -------
        A ready to use transformer.
        """
        config = self.config
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        return materialize(self.name, self.path, config)

    @property
    @lru_cache()
    def serialized(self):
        return serialize(self)

    @staticmethod
    def _to_config(locals_):
        """Convert a dict coming from locals() to config."""
        config = locals_.copy()
        del config['self']
        del config['__class__']
        return config

    def _uuid_impl00(self):
        return self.serialized

    @classproperty
    @lru_cache()
    def name(cls):
        return cls.__name__

    @property
    @lru_cache()
    def longname(self):
        return self.name

    @classproperty
    @lru_cache()
    def path(cls):
        return cls.__module__

    @property
    @lru_cache()
    def wrapped(self):
        """Same as unwrap(), but with the external container Wrap."""
        return None

    @property
    @lru_cache()
    def unwrap(self):
        """Subpipeline inside the first Wrap().

        Hopefully there is only one Wrap in the pipeline.
        This method performs a depth-first search.

        Example:
        pipe = Pipeline(
            File(name='iris.arff'),
            Wrap(Std(), SVMC()),
            Metric(function='accuracy')
        )
        pipe.unwrap  # -> Chain(Std(), SVMC())
        """
        return self.wrapped.transformer


class ITransformer(UTransformer, ABC):
    from pjdata.specialdata import NoData

    def _use_impl(self, data):
        """Each component should implement its core 'apply' functionality."""
        return data

    def apply(self, data: Union[type, Data] = NoData, exit_on_error=True):
        if data.isfrozen:
            return Model(self, data, data)
        if data.allfrozen:
            return Model(self, data, data.frozen)

        self._check_nodata(data, self)

        # Disable warnings, measure time and make the party happen.
        self._handle_warnings()  # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        start = self._cpu()
        try:
            # Aqui, passa-se _exit_on_error para self de forma que
            # implementadores de conteineres possam acessar o valor
            # dentro de
            # _apply_impl e repassar aos contidos. TODO: Mesmo p/ max_time?
            self._exit_on_error = exit_on_error

            model = self._limit_by_time(
                function=self._apply_impl,
                data=data,
                max_time=self.max_time
            )

            # Check result type.
            if not isinstance(model, Model):
                raise Exception(f'{self.name} does not handle {type(model)}!')
        except Exception as e:
            self._handle_exception(e, exit_on_error)
            applied = data.updated(
                self.transformations('a'), failure=str(e), frozen=True
            )
            model = Model(self, data, applied)
            # TODO: é possível que um container não complete o try acima?
            #  Caso sim, devemos gerar um ContainerModel aqui?

        # TODO: put time_spent inside data (as a "volatile" matrix)?
        time_spent = self._cpu() - start
        self._dishandle_warnings()  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self._check_history(data, model.data, self.transformations('a'))
        return model


class INSTransformer(ITransformer, ABC):
    """Independent and NonSymmetric (INS) Transformer.

    In this type of transformer, the 'apply' method  is not dependent
    of the 'use'. It can run without the preavious use of the 'apply' method.
    """
    @abstractmethod
    def _apply_impl(self, data):
        pass

    @abstractmethod
    def _use_impl(self, data):
        pass


class ISTransformer(ITransformer, ABC):
    """Independent and Symmetric (IS) Transformer.

    In this type of transformer, the 'apply' method is not dependent
    of the 'use'. It can run without the preavious use of the 'apply' method.
    """
    def _apply_impl(self, data):
        # TODO: implement the Container Model Case
        # I think that a container model came aways from DTransfomer ...
        after_data = self._use_impl(data, step='a')
        return Model(self, data, after_data)

    @abstractmethod
    def _use_impl(self, data, step='u'):
        pass


class DTransformer(UTransformer, ABC):
    from pjdata.specialdata import NoData

    @abstractmethod
    def _apply_impl(self, data):
        pass

    @abstractmethod
    def _use_impl(self, data, **kwargs):
        pass

    def apply(self, data: Union[type, Data] = NoData, exit_on_error=True):
        if data.isfrozen:
            return EarlyEndedModel(self, data, data)
        if data.allfrozen:
            return EarlyEndedModel(self, data, data.frozen)
        # TODO: We should check if this 'if' is necessary, because when a model
        #  fail a FailedModel with frozen data is returned
        if data.failure:
            return FailedModel(self, data, data)

        self._check_nodata(data, self)

        # Disable warnings, measure time and make the party happen.
        self._handle_warnings()  # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        start = self._cpu()
        try:
            # Aqui, passa-se _exit_on_error para self de forma que
            # implementadores de conteineres possam acessar o valor
            # dentro de
            # _apply_impl e repassar aos contidos. TODO: Mesmo p/ max_time?
            self._exit_on_error = exit_on_error

            model = self._limit_by_time(
                function=self._apply_impl,
                data=data,
                max_time=self.max_time
            )

            # Check result type.
            if not isinstance(model, Model):
                raise Exception(f'{self.name} does not handle {type(model)}!')
        except Exception as e:
            self._handle_exception(e, exit_on_error)
            applied = data.updated(
                self.transformations('a'), failure=str(e), frozen=True
            )
            self._check_history(data, applied, self.transformations('a'))
            return FailedModel(self, data, applied)
            # TODO: é possível que um container não complete o try acima?
            #  Caso sim, devemos gerar um ContainerModel aqui?

        # TODO: put time_spent inside data (as a "volatile" matrix)?
        time_spent = self._cpu() - start
        self._dishandle_warnings()  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self._check_history(data, model.data, self.transformations('a'))
        return model


class NTransformer(Printable, Identifyable, ExceptionHandler, Timers, ABC):

    def __init__(self, config,
                 deterministic=False, nodata_handler=False, max_time=3600):
        jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
        Printable.__init__(self, jsonable)

        self.config = config
        self.deterministic = deterministic
        from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
        self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler
        self._exit_on_error = True
        self.cs = self.cs1  # Shortcut to ease retrieving a CS from a
        # Transformer object without having to check that it is not a
        # component (Transformer class).

        self.max_time = max_time  # TODO: who/when to define maxtime?

    @abstractmethod
    def _use_impl(self, data):
        """Each component should implement its core 'apply' functionality."""

    @abstractmethod
    def apply(self, data, exit_on_error=False):
        """Training step (usually).

         Fit/remove-noise-from/evaluate/... Data.

         Implementers: If your component requires apply() before use(),
         it should extend mixin EnforceApply and _apply_impl() should
         return (None, use_impl()) when provided data is None.

         Parameters
         ----------
         data
             'None' means 'pipeline ended before this transformation'.
             'NoData' means 'pipeline alive, hoping to generate Data in
             the next transformer'.
         exit_on_error
             Exit imediatly instead of just marking a failure inside Data
             object.

         Returns
         -------
         Transformed data, normally.
         None, when data is a None
             (probably meaning the pipeline finished before this transformer).
         Same data, but annotated with a failure.

         Exception
         ---------
         BadComponent
             Data resulting history should be consistent with
             _transformations() implementation.
         """

    @abstractmethod
    def _apply_impl(self, data):
        """Each component should implement its core 'apply' functionality."""

    @classmethod
    @abstractmethod
    def _cs_impl(cls):
        """Each component should implement its own 'cs'. The parent class
        takes care of 'name' and 'path' arguments of ConfigSpace"""

    # TODO: Is unbounded lrucache a source of memory leak?
    @lru_cache()
    def transformations(self, step, clean=True):
        """Expected transformation described as a list of Transformation
        objects.

        Child classes should override this method to perform non-atomic or
        non-trivial transformations.
        A missing implementation will be detected during apply/use."""
        if step in 'au':
            return [Transformation(self, step)]
        else:
            raise BadComponent('Wrong current step:', step)

    @classproperty
    @lru_cache()
    def cs(cls):
        """Config Space of this component, when called as class method.
        If called on an transformer (object/instance method), will convert
        the object to a config space with a single transformer.

        Each Config Space is a tree, where each path represents a parameter
        space of the learning/processing/evaluating algorithm of this component.
        It is a possibly infinite set of configurations.

        Returns
        -------
            Tree representing all the possible parameter spaces.
        """
        cs_ = cls._cs_impl()
        return cs_.identified(name=cls.__name__, path=cls.__module__)

    @property
    @lru_cache()
    def cs1(self=None):
        """Convert transformer into a config space with a single transformer
        inside it."""
        return ConfigList(self)

    def updated(self, **kwargs):
        """Clone this transformer, optionally replacing given params.

        Returns
        -------
        A ready to use transformer.
        """
        config = self.config
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        return materialize(self.name, self.path, config)

    @property
    @lru_cache()
    def serialized(self):
        return serialize(self)

    @staticmethod
    def _to_config(locals_):
        """Convert a dict coming from locals() to config."""
        config = locals_.copy()
        del config['self']
        del config['__class__']
        return config

    def _uuid_impl00(self):
        return self.serialized

    @classproperty
    @lru_cache()
    def name(cls):
        return cls.__name__

    @property
    @lru_cache()
    def longname(self):
        return self.name

    @classproperty
    @lru_cache()
    def path(cls):
        return cls.__module__

    @property
    @lru_cache()
    def wrapped(self):
        """Same as unwrap(), but with the external container Wrap."""
        return None

    @property
    @lru_cache()
    def unwrap(self):
        """Subpipeline inside the first Wrap().

        Hopefully there is only one Wrap in the pipeline.
        This method performs a depth-first search.

        Example:
        pipe = Pipeline(
            File(name='iris.arff'),
            Wrap(Std(), SVMC()),
            Metric(function='accuracy')
        )
        pipe.unwrap  # -> Chain(Std(), SVMC())
        """
        return self.wrapped.transformer
