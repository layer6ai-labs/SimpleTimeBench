"""
Top-level package entry point for ``ts_toolkit``.

The main forecaster classes are exposed lazily to avoid circular-import issues
for callers that only need utilities deeper in the package (e.g., pipeline
helpers).  Accessing the exported names will trigger the required imports.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["AsyncTimeCopilot", "TimeCopilot", "TimeCopilotForecaster"]

_ATTR_MODULE_MAP = {
    "AsyncTimeCopilot": ("ts_toolkit.pipeline.agent", "AsyncTimeCopilot"),
    "TimeCopilot": ("ts_toolkit.pipeline.agent", "TimeCopilot"),
    "TimeCopilotForecaster": (
        "ts_toolkit.pipeline.forecaster",
        "TimeCopilotForecaster",
    ),
}


def __getattr__(name: str):
    if name not in _ATTR_MODULE_MAP:
        raise AttributeError(f"module 'ts_toolkit' has no attribute '{name}'")

    module_name, attr_name = _ATTR_MODULE_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
