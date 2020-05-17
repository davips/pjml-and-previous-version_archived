from pjml.config.description.cs.abc.operatorcs import OperatorCS


class ChainCS(OperatorCS):
    """A Chain is sampled."""

    def sample(self):
        transformers = [cs.sample() for cs in self.components]
        from pjml.tool.chain import Chain
        return Chain(transformers=transformers)


class TChainCS(OperatorCS):
    """A Chain is sampled."""

    def sample(self):
        transformers = [cs.sample() for cs in self.components]
        from pjml.tool.chain import TChain
        return TChain(transformers=transformers)
