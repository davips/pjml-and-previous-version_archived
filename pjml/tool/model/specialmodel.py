from pjml.tool.model.model import Model


class FailedModel(Model):
    def __init__(self, transformer, data_before_apply, data_after_apply,
                 **kwargs):
        super().__init__(transformer, data_before_apply, data_after_apply,
                         **kwargs)

    def _use_impl(self, data, **kwargs):
        raise Exception(
            f"A {self.name} model from failed pipelines during apply is not "
            f"usable!"
        )


class EarlyEndedModel(Model):
    def __init__(self, transformer, data_before_apply, data_after_apply,
                 **kwargs):
        super().__init__(transformer, data_before_apply, data_after_apply,
                         **kwargs)

    def _use_impl(self, data, **kwargs):
        raise Exception(
            f"A {self.name} model from early ended pipelines during apply is "
            f"not usable!"
        )


class CachedApplyModel(Model):
    def __init__(self, transformer, data_before_apply, data_after_apply,
                 **kwargs):
        super().__init__(transformer, data_before_apply, data_after_apply,
                         **kwargs)

    def _use_impl(self, data, **kwargs):
        raise Exception(
            f"A {self.name} model from a succesfully cached apply is not "
            f"usable!"
        )
