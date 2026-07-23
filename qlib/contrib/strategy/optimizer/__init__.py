# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base import BaseOptimizer
from .optimizer import PortfolioOptimizer

try:
    from .enhanced_indexing import EnhancedIndexingOptimizer
except ModuleNotFoundError as e:
    if str(getattr(e, "name", "")) != "numpy.lib.array_utils":
        raise
    EnhancedIndexingOptimizer = None


__all__ = ["BaseOptimizer", "PortfolioOptimizer", "EnhancedIndexingOptimizer"]
