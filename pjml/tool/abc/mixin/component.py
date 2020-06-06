""" Component module. """
from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Dict, Any, Callable, Tuple, Iterator, List, Optional

import pjdata.types as t

from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.aux.util import Property
from pjdata.content.collection import Collection

from pjdata.aux.uuid import UUID
from pjdata.content.content import Content
from pjdata.mixin.identifiable import Identifiable
from pjdata.mixin.printable import Printable
from pjdata.transformer import Transformer
from pjml.config.description.cs.configlist import ConfigList
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.tool.abc.mixin.exceptionhandler import BadComponent


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

        self.cs = self.cs1  # TODO: This can take some time to type.
        self._enhance = enhance
        self._model = model

    @Property
    def jsonable(self):
        return self._jsonable

    @abstractmethod
    def _enhancer_info(self, data: Optional[t.Data] = None) -> Dict[str, Any]:  # <-- TODO: check this optional
        pass

    @abstractmethod
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _enhancer_func(self) -> Callable[[t.Data], t.Data]:
        pass

    @abstractmethod
    def _model_func(self, data: t.Data) -> Callable[[t.Data], t.Data]:
        pass

    def iterators(
            self,
            prior_collection: Collection,
            posterior_collection: Collection
    ) -> Tuple[Iterator, Iterator]:
        raise Exception('NotImplementedError: Only concurrent components have '
                        'iterators')

    @Property
    @lru_cache()
    def enhancer(self) -> Transformer:  # clean, cleaup, dumb, dumb_transformer
        if not self._enhance:
            return Transformer(self.uuid, None, None)
        return Transformer(self, func=self._enhancer_func(), info=self._enhancer_info)

    # TODO: verify if Data (/ Collection?) should have a better __hash__
    @lru_cache()
    def model(self, prior: Content) -> Transformer:  # <-- com os novos tipos ficou mais curto, então subi
        if isinstance(prior, tuple): # <-- Pq??
            prior = prior[0]
        if not self._model:
            return Transformer(self.uuid, None, None)
        # Assumes all components are symmetric. I.e. we can use the same self for both enhance and model.
        return Transformer(self, func=self._model_func(prior), info=self._model_info(prior))

    # TODO: special sub class for concurrent components containing the content
    #   of this IF and the parent ABC method iterator().
    def dual_transform(self,
                       prior: t.DataOrColl,
                       posterior: t.DataOrCollOrTup
                       ) -> Tuple[t.DataOrColl, t.DataOrCollOrTup]: #TODO: overload

        # We need to put the ignore here because @porperty has not annotations.
        # Another alternative is creating our own @property decorator and
        # putting Any as a return. More information can be found on mypy's
        # Github, issue #1362
        prior_result = self.enhancer.transform(prior)
        posterior_result = self.model(prior).transform(posterior)
        return prior_result, posterior_result

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
        print('TODO: aproveitar processamento do cfg_serialized!')  # <-- TODO
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
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        return materialize(self.name, self.path, config)
