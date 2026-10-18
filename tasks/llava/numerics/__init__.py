"""Standard precision evaluation: record -> calibrate -> compare (+ fp64 arbitration) -> selftest -> TOST.

Tasks only implement numerics.adapter.Adapter.
"""
from .adapter import Adapter, Objects

__all__ = ['Adapter', 'Objects']
