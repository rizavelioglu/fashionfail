import importlib.metadata

__modulename__: str = __name__.split(".")[0]
__version__ = importlib.metadata.version(__modulename__)
