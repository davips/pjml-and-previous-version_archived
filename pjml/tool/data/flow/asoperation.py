from pjml.tool.abc.minimalcontainer import MinimalContainer1


class AlwaysApply(MinimalContainer1):
    """Always 'apply' regardless of current step."""

    # TODO: implement __new__ to generate a CS

    def _apply_impl(self, data):
        self.model = NoModel
        return self.component.apply(data)

    def _use_impl(self, data, **kwargs):
        return self.component.apply(data)


class AlwaysUse(MinimalContainer1):
    """Always 'use' regardless of current step."""

    # TODO: implement __new__ to generate a CS

    def _apply_impl(self, data):
        self.model = NoModel
        return self.component.use(data)

    def _use_impl(self, data, **kwargs):
        return self.component.use(data)
