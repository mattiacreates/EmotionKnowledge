class Tool:
    """Minimal tool wrapper providing an ``invoke`` method."""

    def __init__(self, func):
        self.func = func
        self.__doc__ = getattr(func, "__doc__")
        self.__name__ = getattr(func, "__name__", "Tool")

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def invoke(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        return self.func(*args, **kwargs)


def tool(func):
    """Return a :class:`Tool` instance for the decorated function."""
    return Tool(func)
