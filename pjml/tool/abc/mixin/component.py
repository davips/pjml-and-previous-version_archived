""" Component module. """
from abc import abstractmethod, ABC
from functools import lru_cache, cached_property
from typing import Dict, Any, Tuple, Iterator

import pjdata.types as t
from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.aux.util import Property
from pjdata.aux.uuid import UUID
from pjdata.mixin.identifiable import Identifiable
from pjdata.mixin.printable import Printable
from pjdata.transformer import Transformer
from pjml.config.description.cs.configlist import ConfigList
from pjml.config.description.cs.transformercs import TransformerCS


class Component(Printable, Identifiable, ABC):
    def __init__(
            self,
            config: dict,
            enhance: bool = True,
            model: bool = True,
            deterministic: bool = False,
            nodata_handler: bool = False
    ):
        self.transformer_info = {'_id': f'{self.name}@{self.path}', 'config': config}
        self._jsonable = {'info': self.transformer_info, 'enhance': enhance, 'model': model}

        self.config = config
        self.deterministic = deterministic

        from pjml.tool.abc.mixin.nodatahandler import NoDataHandler
        self.nodata_handler = isinstance(self, NoDataHandler) or nodata_handler

        self.cs = self.cs1  # TODO: This can take some time to type. It is pure magic!
        self._enhance = enhance
        self._model = model

    @Property
    def jsonable(self):
        return self._jsonable

    @abstractmethod
    def _enhancer_info(self, data: t.Data) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _enhancer_func(self) -> t.Transformation:
        pass

    @abstractmethod
    def _model_func(self, data: t.Data) -> t.Transformation:
        pass

    @Property
    @lru_cache()
    def enhancer(self) -> Transformer:
        if not self._enhance:
            return Transformer(self, None, None)
        return Transformer(self, func=self._enhancer_func(), info=self._enhancer_info)

    # TODO: verify if Data (/ Collection?) should have a better __hash__
    @lru_cache()
    def model(self, data: t.Data) -> Transformer:
        if isinstance(data, tuple):  # <-- Pq??
            data = data[0]
        if not self._model:
            return Transformer(self, None, None)
        # Assumes all components are symmetric. I.e. we can use the same self for both enhance and model.
        return Transformer(self, func=self._model_func(data), info=self._model_info(data))

    # TODO: special sub class for concurrent components containing the content
    #   of this IF and the parent ABC method iterator().
    def dual_transform(self, train: t.Data, test: t.Data) -> Tuple[t.Data, t.Data]:

        # We need to put the ignore here because @porperty has not annotations.
        # Another alternative is creating our own @property decorator and
        # putting Any as a return. More information can be found on mypy's
        # Github, issue #1362
        if self._model:  # TODO: I am not sure these IFs are really needed...
            test = self.model(train).transform(test)
        if self._enhance:  # TODO: ... I've put them here because of streams.
            train = self.enhancer.transform(train)
        return train, test

    @classmethod
    @abstractmethod
    def _cs_impl(cls) -> TransformerCS:
        """Each component should implement its own 'cs'. The parent class
        takes care of 'name' and 'path' arguments of ConfigSpace"""

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

    @Property
    @lru_cache()
    def cs1(self=None):
        """Convert transformer into a config space with a single transformer
        inside it."""
        return ConfigList(self)

    @Property
    @lru_cache()
    def serialized(self):
        # print('TODO: aproveitar processamento do cfg_serialized!')  # <-- TODO
        return serialize(self)

    @staticmethod
    def _to_config(locals_):
        """Convert a dict coming from locals() to config."""
        config = locals_.copy()
        del config['self']
        del config['__class__']
        if 'kwargs' in config:
            del config['kwargs']
        if 'enhance' in config:
            del config['enhance']
            del config['model']
        return config

    def _uuid_impl(self):
        """Complete UUID; including 'model' and 'enhance' flags. Identifies the component."""
        return self.cfg_uuid * UUID(str(self._enhance + self._model).rjust(14, '0'))

    @Property
    @lru_cache()
    def cfg_serialized(self):
        return serialize(self.transformer_info)

    @Property
    @lru_cache()
    def cfg_uuid(self):
        """UUID excluding 'model' and 'enhance' flags. Identifies the transformer."""
        return UUID(self.cfg_serialized.encode())

    @classproperty
    @lru_cache()
    def name(cls):
        return cls.__name__

    @Property
    @lru_cache()
    def longname(self):
        return self.name

    @classproperty
    @lru_cache()
    def path(cls):
        return cls.__module__

    @Property
    @lru_cache()
    def wrapped(self):
        """Same as unwrap(), but with the external container Wrap."""
        return None

    @Property
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
        if 'model' not in self.config:
            config.update({'model': self._model})

        if 'enhancer' not in self.config:
            config.update({'enhance': self._enhance})

        if kwargs:
            config = config.copy()
            config.update(kwargs)

        self.disable_pretty_printing()
        # print('OBJ: ', self)
        # print('CONFIG: ', config)

        return materialize(self.name, self.path, config)

    # # TODO: Is unbounded lrucache a source of memory leak?
    # @lru_cache()
    # def transformations(self, step: str, clean: bool = True) -> List[Transformation]:
    #     """Expected transformation described as a list of Transformation
    #     objects.
    #
    #     Child classes should override this method to perform non-atomic or
    #     non-trivial transformations.
    #     A missing implementation will be detected during apply/use.
    #
    #     Parameters
    #     ----------
    #     step: str
    #         TODO
    #     clean: bool
    #         TODO
    #
    #     Returns
    #     -------
    #         list of Transformation
    #     """
    #     if step in "au":
    #         return [Transformation(self, step)]
    #     else:
    #         raise BadComponent("Wrong current step:", step)
