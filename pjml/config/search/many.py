"""Operations over many CSs."""
from pjml.config.description.cs.selectcs import SelectCS
from pjml.config.description.cs.shufflecs import ShuffleCS


def select(*components):
    return SelectCS(*components)


def shuffle(*components):
    return ShuffleCS(*components)
