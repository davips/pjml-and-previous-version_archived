class TComponent:
    def __init__(self, config, deterministic):
        self.config = config
        self.deterministic = deterministic

    def enhancer(self):
        return TTransformer()

    def modeler(self, prior):
        return TTransformer()

    def dual_transform(self, prior, posterior):
        prior_result = self.enhancer().transform(prior)
        posterior_result = self.modeler(prior).transform(posterior)

        return prior_result, posterior_result


class TTransformer:
    def __init__(self, func=None, **kwargs):
        self.func = TTransformer.does_nothing if None else func

    def transform(self, data):
        return self.func(data)

    @staticmethod
    def does_nothing(data): # melhorar o nome
        return data

