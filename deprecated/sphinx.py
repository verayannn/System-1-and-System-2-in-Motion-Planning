from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


def deprecated(*_args, **_kwargs):
    def wrapper(func: F) -> F:
        return func

    return wrapper
