"""
Qualitative Reasoning Modules Package

This package contains modularized components of the qualitative reasoning system,
broken down from the monolithic qualitative_reasoning.py file for better
maintainability, testability, and extensibility.

Modules:
- core_types: Basic data structures and enums
- constraint_engine: Constraint evaluation and safety systems  
- process_engine: Process management and causal reasoning
- simulation_engine: Qualitative simulation and state management
- analysis_engine: Relationship analysis and behavior explanation
- visualization_engine: System state visualization and reporting
- safety_config: Configuration management for safe constraint evaluation

Author: Benedict Chen
"""

__version__ = "1.0.0"

from .core_types import (
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

# Constraint engine - safety-critical constraint evaluation
from .constraint_engine import (
    ConstraintEvaluationMethod,
    ConstraintEvaluationConfig,
    ConstraintEngineMixin
)
# from .process_engine import ProcessEngine
# from .simulation_engine import SimulationEngine  
# from .analysis_engine import AnalysisEngine
# from .visualization_engine import VisualizationEngine
# from .safety_config import SafetyConfigManager

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
    "QProcess",
    
    # Constraint engine components
    "ConstraintEvaluationMethod",
    "ConstraintEvaluationConfig", 
    "ConstraintEngineMixin",
    
    # Note: Other components will be added as they are implemented
    # "ProcessEngine",
    # "SimulationEngine",
    # "AnalysisEngine", 
    # "VisualizationEngine",
    # "SafetyConfigManager"
]