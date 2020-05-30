import operator
from abc import abstractmethod, ABC
from functools import lru_cache
from itertools import tee

from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.collection import Collection
from pjdata.mixin.identifyable import Identifyable
from pjdata.mixin.printable import Printable
from pjdata.step.transformation import Transformation
from pjml.config.description.cs.configlist import ConfigList
from pjml.tool.abc.mixin.exceptionhandler import BadComponent


class TComponent(Printable, Identifyable, ABC):
    def __init__(self, config, onenhancer=True, onmodel=True,
                 deterministic=False, nodata_handler=False):
        jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
        Printable.__init__(self, jsonable)

        self.config = config
        self.deterministic = deterministic

        from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
        self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler

        self.cs = self.cs1
        self.onenhancer = onenhancer
        self.onmodel = onmodel

    def _enhancer_impl(self):
        return TTransformer(None, None)

    def _model_impl(self, prior):
        return TTransformer(None, None)

    def iterators(self, prior_collection, posterior_collection):
        raise NotImplementedError('Only concurrent components have iterators')

    @property
    @lru_cache()
    def enhancer(self):  # clean, cleaup, dumb, dumb_transformer
        if not self.onenhancer:
            return TTransformer(None, None)
        return self._enhancer_impl()

    # TODO: verify if Data (/ Collection?) should have a better __hash__
    @lru_cache()
    def model(self, prior):  # smart, smart_transformer
        if isinstance(prior, tuple):
            prior = prior[0]
        if not self.onmodel:
            return TTransformer(None, None)
        return self._model_impl(prior)

    # TODO: special sub class for concurrent components containing the content
    #   of this IF and the parent ABC method iterator().
    def dual_transform(self, prior, posterior):
        if isinstance(prior, Collection) or isinstance(posterior, Collection):
            iterator1, iterator2 = self.iterators(prior, posterior)
            if self.onenhancer:
                prior = Collection(iterator1, lambda: prior.data,
                                   debug_info='compo' + self.__class__.__name__ + ' pri')
            if self.onmodel:
                posterior = Collection(iterator2, lambda: posterior.data,
                                       debug_info='compo' + self.__class__.__name__ + ' pos')
            return prior, posterior

        prior_result = self.enhancer.transform(prior)
        posterior_result = self.model(prior).transform(posterior)
        return prior_result, posterior_result

    @classmethod
    @abstractmethod
    def _cs_impl(cls):
        """Each component should implement its own 'cs'. The parent class
        takes care of 'name' and 'path' arguments of ConfigSpace"""

    # # TODO: Is unbounded lrucache a source of memory leak?
    # @lru_cache()
    # def transformations(self, step, clean=True):
    #     """Expected transformation described as a list of Transformation
    #     objects.
    #
    #     Child classes should override this method to perform non-atomic or
    #     non-trivial transformations.
    #     A missing implementation will be detected during apply/use."""
    #     if step in 'au':
    #         return [Transformation(self, step)]
    #     else:
    #         raise BadComponent('Wrong current step:', step)

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
        result = cs_.identified(name=cls.__name__, path=cls.__module__)
        return result

    @property
    @lru_cache()
    def cs1(self=None):
        """Convert transformer into a config space with a single transformer
        inside it."""
        return ConfigList(self)

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
        if 'kwargs' in config:
            del config['kwargs']
        if 'onenhancer' in config:
            del config['onenhancer']
            del config['onmodel']
        return config

    def _uuid_impl(self):
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


class TTransformer:
    def __init__(self, func, info):
        self.func = func if func else lambda data: data
        self._info = info

        # Note:
        # Callable returns True, if the object appears to be callable
        # Yes, that appears!
        if callable(self._info):
            self.info = self._info
        elif isinstance(self._info, dict):
            self.info = lambda: self._info
        elif self._info is None:
            self.info = {}
        else:
            raise TypeError('Unexpected info type. You should use, callable, '
                            'dict or None.')

    def transform(self, data):  # resolver error
        # print('!!!!!!!!!!!!!!!', type(self).__name__, type(data))
        if isinstance(data, tuple):
            return tuple((self.safe_func(dt) for dt in data))
        # Todo: We should add exception handling here because self.func can
        #  raise an error
        return self.safe_func(data)

    def safe_func(self, data):
        if data.isfrozen:
            return data
        return self.func(data)
