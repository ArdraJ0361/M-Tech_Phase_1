"""
src/data/__init__.py

Note: imports are deferred to avoid requiring datasets/transformers at import time.
"""
from .dataset_loader import load_dataset_for_task  # noqa: F401
