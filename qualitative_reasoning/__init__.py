"""
Qualitative Reasoning Package

A comprehensive Python library for qualitative reasoning systems based on
Forbus's Process Theory and de Kleer's Qualitative Physics framework.

This package enables AI systems to reason about physical systems using
qualitative relationships rather than precise numerical values, similar
to how humans understand physics intuitively.

Modules:
- qr_modules: Modularized components of the qualitative reasoning system

Author: Benedict Chen
"""

__version__ = "1.0.0"

# Import core types and utilities for easy access
from .qr_modules.core_types import (
    QualitativeValue,
    QualitativeDirection,
    QualitativeQuantity,
    QualitativeState, 
    QualitativeProcess,
    # Utility functions
    compare_qualitative_values,
    qualitative_to_numeric,
    numeric_to_qualitative,
    create_quantity,
    validate_qualitative_state,
    # Type aliases
    QValue,
    QDirection,
    QQuantity,
    QState,
    QProcess
)

__all__ = [
    # Core types
    "QualitativeValue",
    "QualitativeDirection", 
    "QualitativeQuantity",
    "QualitativeState",
    "QualitativeProcess",
    
    # Utility functions
    "compare_qualitative_values",
    "qualitative_to_numeric",
    "numeric_to_qualitative", 
    "create_quantity",
    "validate_qualitative_state",
    
    # Type aliases
    "QValue",
    "QDirection",
    "QQuantity", 
    "QState",
    "QProcess"
]