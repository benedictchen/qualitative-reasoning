"""
🧠 Qualitative Reasoning - How AI Understands Physics Like Humans
===============================================================

📚 Research Papers:
Forbus, K. D., & de Kleer, J. (1993)
"Building Problem Solvers"
MIT Press

de Kleer, J., & Brown, J. S. (1984)
"A Qualitative Physics Based on Confluences"
Artificial Intelligence, 24(1-3), 7-83

🎯 ELI5 Summary:
Imagine trying to understand a bathtub filling with water without knowing exact numbers.
You know: water flows in → level rises → might overflow. This is qualitative reasoning!
It's how humans understand physics - through cause and effect relationships, not equations.
AI can use this to understand physical systems like a smart human would.

🧪 Research Background:
Traditional AI used precise mathematical models requiring exact numerical values.
Forbus & de Kleer revolutionized this by showing AI could reason about physical
systems using qualitative relationships:

Key breakthroughs:
- Qualitative differential equations without numbers
- Confluence-based causal reasoning  
- Envisionment generation for behavior prediction
- Human-like understanding of physical systems

🔬 Mathematical Framework:
Qualitative State: [Q-value, Q-direction]
- Q-value ∈ {-, 0, +} (negative, zero, positive)
- Q-direction ∈ {inc, std, dec} (increasing, steady, decreasing)

Confluences: IF condition THEN consequence
Envisionment: All possible qualitative behaviors from initial state

🎨 ASCII Diagram - Qualitative Water Tank:
========================================

    Input Flow    │    Qualitative States:
        ▼         │    ┌─────────────────┐
    ┌─────────┐   │    │ Level: [+, inc] │ ← Rising
    │  TANK   │   │    │ Flow:  [+, std] │ ← Constant inflow  
    │ ░░░░░░░ │   │    │ Time:  [+, inc] │ ← Always increasing
    │ ░░Water░ │   │    └─────────────────┘
    │ ░░░░░░░ │   │    
    └─────────┘   │    Confluence Rules:
        ▼         │    IF inflow > outflow 
    Output Flow   │    THEN level increases

🏗️ Implementation Features:
✅ Qualitative differential equations
✅ Confluence-based constraint propagation
✅ Envisionment generation algorithms
✅ Safe constraint evaluation (no eval() security risks)
✅ Physical system modeling
✅ Causal reasoning chains

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

🔗 Related Work: Causal Reasoning, Physics Simulation, Knowledge Representation
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import ast
import operator
import re
warnings.filterwarnings('ignore')


class QualitativeValue(Enum):
    """Qualitative values for continuous quantities"""
    NEGATIVE_INFINITY = "neg_inf"
    DECREASING_LARGE = "dec_large"
    NEGATIVE_LARGE = "neg_large"
    DECREASING = "decreasing"
    NEGATIVE_SMALL = "neg_small"
    ZERO = "zero"
    POSITIVE_SMALL = "pos_small"
    INCREASING = "increasing"
    POSITIVE_LARGE = "pos_large"
    INCREASING_LARGE = "inc_large"
    POSITIVE_INFINITY = "pos_inf"


class QualitativeDirection(Enum):
    """Qualitative directions for change"""
    INCREASING = "+"
    DECREASING = "-"
    STEADY = "0"
    UNKNOWN = "?"


class ConstraintEvaluationMethod(Enum):
    """Methods for evaluating constraints safely"""
    UNSAFE_EVAL = "unsafe_eval"        # Original eval() method (NOT RECOMMENDED)
    AST_SAFE = "ast_safe"             # Safe AST-based evaluation
    REGEX_PARSER = "regex_parser"     # Regular expression parsing
    CSP_SOLVER = "csp_solver"         # Constraint Satisfaction Problem solver
    HYBRID = "hybrid"                 # Combine multiple methods


@dataclass 
class ConstraintEvaluationConfig:
    """Configuration for constraint evaluation with maximum safety and flexibility"""
    
    # Primary evaluation method
    evaluation_method: ConstraintEvaluationMethod = ConstraintEvaluationMethod.AST_SAFE
    
    # Safety settings
    allow_function_calls: bool = False
    allow_attribute_access: bool = False
    allowed_operators: Set[str] = None  # Default will be set to safe operators
    allowed_names: Set[str] = None      # Variables allowed in expressions
    
    # Parser settings
    enable_regex_fallback: bool = True
    enable_type_checking: bool = True
    
    # CSP solver settings 
    csp_solver_backend: str = "backtracking"  # "backtracking", "arc_consistency"
    csp_timeout_ms: int = 1000
    
    # Error handling
    strict_mode: bool = False  # If True, fail on any parsing error
    fallback_to_false: bool = True  # If evaluation fails, assume constraint violated
    
    def __post_init__(self):
        if self.allowed_operators is None:
            self.allowed_operators = {
                'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow',
                'Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq', 
                'And', 'Or', 'Not', 'Is', 'IsNot', 'In', 'NotIn'
            }
        if self.allowed_names is None:
            self.allowed_names = set()  # Will be populated with quantity names


@dataclass
class QualitativeQuantity:
    """Represents a quantity with qualitative magnitude and direction"""
    name: str
    magnitude: QualitativeValue
    direction: QualitativeDirection
    landmark_values: Optional[List[float]] = None  # Critical values


@dataclass
class QualitativeState:
    """Complete qualitative state of a system at one instant"""
    time_point: str
    quantities: Dict[str, QualitativeQuantity]
    relationships: Dict[str, str]  # Derived relationships


@dataclass
class QualitativeProcess:
    """Represents a process with preconditions, quantity conditions, and influences"""
    name: str
    preconditions: List[str]  # What must be true
    quantity_conditions: List[str]  # Constraints on quantities  
    influences: List[str]  # How this process affects quantities
    active: bool = False


class QualitativeReasoner:
    """
    Qualitative Reasoning System following Forbus's Process Theory
    and de Kleer's Qualitative Physics framework
    
    The key insight: Physical understanding comes from reasoning about
    qualitative relationships and processes, not precise numbers.
    
    Core principles:
    1. Quantities have qualitative values (zero, positive, negative) 
    2. Processes influence quantities over time
    3. Constraints maintain consistency
    4. Landmark values create discrete behavior regions
    """
    
    def __init__(self, domain_name: str = "Generic Physical System", 
                 constraint_config: Optional[ConstraintEvaluationConfig] = None):
        """
        Initialize Qualitative Reasoner with configurable constraint evaluation
        
        Args:
            domain_name: Name of the physical domain being modeled
            constraint_config: Configuration for safe constraint evaluation
        """
        
        self.domain_name = domain_name
