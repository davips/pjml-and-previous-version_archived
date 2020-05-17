from pjml.tool.chain import Chain, TChain


class Pipeline(Chain):
    """Chain the execution of the given transformers."""


class TPipeline(TChain):
    """Chain the execution of the given transformers."""
