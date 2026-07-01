from __future__ import annotations

from functools import wraps


def deprecated(*dargs, **dkwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if dargs and callable(dargs[0]) and not dkwargs:
        return decorator(dargs[0])
    return decorator

