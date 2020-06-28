from abc import ABC
from functools import lru_cache

from pjdata.aux.decorator import classproperty
from pjdata.aux.util import Property
from pjml.tool.abs.component import Component


class Container(Component, ABC):
    """A container modifies 'component(s)'."""

    def __init__(self, config, seed, components, enhance, model, deterministic):
        if not components:
            raise Exception(
                f"A container ({self.name}) should have at least one " f"component!"
            )

        # components=[Chain(A)] should appear as components=[A] in config.
        from pjml.tool.chain import Chain

        if len(components) == 1 and isinstance(components[0], Chain):
            components = components[0].components

        # Propagate seed.
        self.components = []
        for component in components:
            kwargs = {}
            if "seed" not in component.config and not component.deterministic:
                kwargs["seed"] = seed

            # if the user sets a Container attribute, this state value is passed
            # forward to the internal Components.
            if not model:
                kwargs["model"] = model

            if not enhance:
                kwargs["enhance"] = enhance

            component = component.updated(**kwargs)
            self.components.append(component)

        complete_config = {"components": self.components}
        complete_config.update(config)
        super().__init__(
            complete_config,
            enhance=enhance,
            model=model,
            deterministic=deterministic,
            nodata_handler=self.components[0].nodata_handler,
        )

    @Property
    @lru_cache()
    def wrapped(self):
        from pjml.tool.meta.wrap import Wrap  # TODO: port this thing to Pipeline approach

        for component in self.components:
            component = component.wrapped
            if isinstance(component, Wrap):
                return component
        return None

    @classmethod
    @classproperty
    def cs(cls):
        raise Exception(
            f"{cls.name} depends on components to build a CS.\n"
            f"Just instantiate the class {cls.name} instead of calling its .cs!"
        )

    @Property
    @lru_cache()
    def longname(self):
        names = ", ".join([tr.longname for tr in self.components])
        return self.name + f"[{names}]"

    @classmethod
    def _cs_impl(cls):
        raise Exception(f"Wrong calling of {cls.name}._cs_impl!")

    def __str__(self, depth=""):
        if not self.pretty_printing:
            return super().__str__()

        inner = []
        for t in self.components:
            inner.append("    " + t.__str__(depth).replace("\n", "\n" + "    "))

        return f"{depth}{self.name}>>\n" + "\n".join(inner) + f"\n{depth}<<{self.name}"
