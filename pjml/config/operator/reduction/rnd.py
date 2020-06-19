from pjml.config.description.cs.configlist import ConfigList


def rnd(cs, n=100):
    """Reduces CS by random sampling."""
    return ConfigList(components=[cs.cs.sample() for _ in range(n)])
