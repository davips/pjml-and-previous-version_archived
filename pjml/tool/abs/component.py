""" Component module. """
from __future__ import annotations

from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Dict, Any, Tuple
from typing import TYPE_CHECKING

from pjdata.mixin.serialization import withSerialization

if TYPE_CHECKING:
    import pjdata.types as t
from pjdata.aux.decorator import classproperty
from pjdata.aux.serialization import serialize, materialize
from pjdata.aux.util import Property
from pjdata.aux.uuid import UUID
from pjdata.mixin.printing import withPrinting
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.cs.configlist import ConfigList
from pjml.config.description.cs.cs import CS
from pjml.tool.abs.asoperand import asOperand


class Component(withPrinting, withSerialization, asOperand, ABC):
    def __init__(
            self,
            config: dict,
            enhancer_cls: type,
            model_cls: type,
            enhance: bool = True,
            model: bool = True,
            deterministic: bool = False,
            nodata_handler: bool = False,  # this flag and the mixin are needed, but I can't recall why... [davi]
    ):
        # We must always obtain the default parameter, because we want to completely identify the transformation.
        self.config = self.default_config()
        self.config.update(config)

        self.path = self.__module__
        self.info_for_transformer = {
            "id": f"{self.name}@{self.path}",
            "config": self.config,
        }
        self._jsonable = {
            "info": self.info_for_transformer,
            "enhance": enhance,
            "model": model,
        }

        self.deterministic = deterministic

        from pjml.tool.abs.mixin.nodatahandling import withNoDataHandling

        self.nodata_handler = isinstance(self, withNoDataHandling) or nodata_handler

        self.hasenhancer = enhance
        self.hasmodel = model
        self.enhancer_cls = enhancer_cls
        self.model_cls = model_cls

        self.cs = self.cs1  # TODO: This can take some time to type. It is pure magic!

    def _jsonable_impl(self):
        return self._jsonable

    @Property
    @lru_cache()
    def enhancer(self) -> Transformer:
        return self.enhancer_cls(self)

    @lru_cache()
    def model(self, data: t.Data) -> Transformer:
        if isinstance(data, tuple):  # <-- Pq??
            data = data[0]
        return self.model_cls(self, data)

    def dual_transform(self, train: t.Data, test: t.DataOrTup) -> Tuple[t.Data, t.DataOrTup]:
        return self.enhancer.transform(train), self.model(train).transform(test)

    @classmethod
    @abstractmethod
    def _cs_impl(cls) -> ConfigSpace:
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
        if "enhancer_cls" in config:
            del config["enhancer_cls"]
            del config["model_cls"]
        return config

    def _uuid_impl(self):
        """Complete UUID; including 'model' and 'enhance' flags. Identifies the component."""
        return self._cfuuid_impl() * UUID(str(self.hasenhancer + self.hasmodel).rjust(14, "0"))

    def _cfuuid_impl(self, data=None):
        """UUID excluding 'model' and 'enhance' flags. Identifies the transformer."""
        return UUID(self._cfserialized().encode())

    @Property
    @lru_cache()
    def cfserialized(self):
        return self._cfserialized()

    def _cfserialized(self):
        return serialize(self.info_for_transformer)

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

        return materialize(self.name, self.path, config)

    # TODO: Is unbounded lrucache a source of memory leak?

    def __lt__(self, other):
        """This is needed because heuristics usually compare tuples (Data, Component), and sometimes,
         there are two identical Data objects, so the heap sorting will attempt to order by the component.
        """
        return False

    def __eq__(self, other):
        return self.uuid == other.uuid

    def __hash__(self):
        return id(self)  # <-- TODO: not optimal, a dict can have many identical components this way.
        # raise Exception
        # return hash(self.uuid)
        # <- TODO: Ideally this should be done by hash(self.uuid), but it gives "Stream not consumed!".
        #        The lrrcaching/property depending on components seems to be the problem here, despite making no sense.
