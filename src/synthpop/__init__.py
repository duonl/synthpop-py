from .synthesiser import Synthesiser

__version__ = "1.0.0"

_submodules = [
    "data_preparation",
    "methods",
    "plotting",
    "utility_metrics",
]

__all__ = _submodules + [
    "Synthesiser",
]
