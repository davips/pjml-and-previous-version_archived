from abc import abstractmethod, ABC
from functools import lru_cache

from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.mixin.identifyable import Identifyable
from pjdata.mixin.printable import Printable
from pjdata.step.transformation import Transformation
from pjml.config.description.cs.configlist import ConfigList
from pjml.tool.abc.mixin.exceptionhandler import BadComponent


class TComponent(Printable, Identifyable, ABC):
    def __init__(self, config, deterministic, nodata_handler=False,):
        jsonable = {'_id': f'{self.name}@{self.path}', 'config': config}
        Printable.__init__(self, jsonable)

        self.config = config
        self.deterministic = deterministic

        from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
        self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler

        self.cs = self.cs1

    def enhancer(self):
        return TTransformer()

    def modeler(self, prior):
        return TTransformer()

    def dual_transform(self, prior, posterior):
        prior_result = self.enhancer().transform(prior)
        posterior_result = self.modeler(prior).transform(posterior)

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
        result= cs_.identified(name=cls.__name__, path=cls.__module__)
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
    def __init__(self, func=None, **kwargs):
        self.func = TTransformer.does_nothing if func is None else func

    def transform(self, data):
        return self.func(data)

    @staticmethod
    def does_nothing(data):  # melhorar o nome
        return data

