from pjml.tool.chain import Chain


# TODO: convert this into a more useful Wrap-like class,
#  e.g. to specify and separate ML steps from the rest of the workflow
class Pipeline(Chain):
    """Chain the execution of the given components."""
