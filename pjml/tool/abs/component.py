""" Component module. """
from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Dict, Any, Tuple

import pjdata.types as t
from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.aux.util import Property
from pjdata.aux.uuid import UUID
from pjdata.mixin.printing import withPrinting
from pjdata.mixin.serialization import WithSerialization
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjdata.transformer.pholder import PHolder
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.configlist import ConfigList
from pjml.config.description.cs.cs import CS
from pjml.tool.abs.asoperand import AsOperand


class Component(withPrinting, WithSerialization, AsOperand, ABC):
    def __init__(
            self,
            config: dict,
            enhance: bool = True,
            model: bool = True,
            deterministic: bool = False,
            nodata_handler: bool = False,
    ):
        # We must always obtain the default parameter, because we want to completely
        # identify the transformation.
        self.config = self.default_config()
        self.config.update(config)

        self.path = self.__module__
        self.transformer_info = {
            "_id": f"{self.name}@{self.path}",
            "config": self.config,
        }
        self._jsonable = {
            "info": self.transformer_info,
            "enhance": enhance,
            "model": model,
        }

        self.deterministic = deterministic

        from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling

        self.nodata_handler = isinstance(self, withNoDataHandling) or nodata_handler

        self.hasenhancer = enhance
        self.hasmodel = model

        self.cs = self.cs1  # TODO: This can take some time to type. It is pure magic!

    def _jsonable_impl(self):
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
        if not self.hasenhancer:
            return PHolder(self)
        return Enhancer(self, func=self._enhancer_func(), info_func=self._enhancer_info)

    @lru_cache()
    def model(self, data: t.Data) -> Transformer:
        if isinstance(data, tuple):  # <-- Pq??
            data = data[0]
        if not self.hasmodel:
            return PHolder(self)
        return Model(
            self, func=self._model_func(data), info=self._model_info(data), data=data
        )

    def dual_transform(self, train: t.Data, test: t.Data) -> Tuple[t.Data, t.Data]:
        # if self._model:  # TODO: I am not sure these IFs are really needed...
        test = self.model(train).transform(test)
        # if self._enhance:  # TODO: ... I've put them here because of streams.
        train = self.enhancer.transform(train)
        return train, test

    @classmethod
    @abstractmethod
    def _cs_impl(cls) -> CS:
        """Each component should implement its own 'cs'. The parent class
        takes care of 'name' and 'path' arguments of ConfigSpace"""

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Create a copy of the component default configuration.

        Returns
        -------
            dict
                Copy of the component default configuration.

        """
        return cls._default_config_impl.copy()

    @classproperty
    def _default_config_impl(cls) -> Dict[str, Any]:
        return {}

    @classproperty
    @lru_cache()
    def cs(cls) -> CS:
        """Config Space of this component, when called as class method.
        If called on a component (object/instance method), will convert
        the object to a config space with a single component.

        Each Config Space is a tree, where each path represents a parameter
        space of the learning/processing/evaluating algorithm of this component.
        It is a possibly infinite set of configurations.

        Returns
        -------
            Tree representing all the possible parameter spaces.
        """
        cs_ = cls._cs_impl()
        # TODO: Why do we send the 'cls' to the CS contructor avoiding to call 'identified' ?
        return cs_.identified(name=cls.__name__, path=cls.__module__)

    @Property
    @lru_cache()
    def cs1(self=None):
        """Convert component into a config space with a single component
        inside it."""
        return ConfigList(self)

    @staticmethod
    def _to_config(locals_):
        """Convert a dict coming from locals() to config."""
        config = locals_.copy()
        del config["self"]
        del config["__class__"]
        if "kwargs" in config:
            del config["kwargs"]
        if "enhance" in config:
            del config["enhance"]
            del config["model"]
        return config

    def _uuid_impl(self):
        """Complete UUID; including 'model' and 'enhance' flags. Identifies the component."""
        return self._cfuuid_impl() * UUID(
            str(self.hasenhancer + self.hasmodel).rjust(14, "0")
        )

    def _cfuuid_impl(self):
        """UUID excluding 'model' and 'enhance' flags. Identifies the transformer."""
        return UUID(self._cfserialized().encode())

    @Property
    @lru_cache()
    def cfserialized(self):
        return self._cfserialized()

    def _cfserialized(self):
        return serialize(self.transformer_info)

    def _name_impl(self):
        return self.__class__.__name__

    @Property
    @lru_cache()
    def longname(self):
        return self.name

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
        return self.wrapped.component

    def updated(self, **kwargs):
        """Clone this component, optionally replacing given params.

        Returns
        -------
        A ready to use component.
        """
        config = self.config
        if "model" not in self.config:
            config.update({"model": self.hasmodel})

        if "enhancer" not in self.config:
            config.update({"enhance": self.hasenhancer})

        if kwargs:
            config = config.copy()
            config.update(kwargs)

        self.disable_pretty_printing()
        # print('OBJ: ', self)
        # print('CONFIG: ', config)

        return materialize(self.name, self.path, config)

    # TODO: Is unbounded lrucache a source of memory leak?

    def __lt__(self, other):
        """This is needed because heuristics usually com pare tuples (Data, Component), and sometimes,
         there are two identical Data objects, so the heap sorting will attempt to order by the component.
        """
        return False
