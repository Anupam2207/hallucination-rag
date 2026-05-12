from functools import lru_cache
from typing import Callable, TypeVar


T = TypeVar("T")


def cached_factory(maxsize: int = 4) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Small wrapper around lru_cache for factory-style helpers."""
    return lru_cache(maxsize=maxsize)
