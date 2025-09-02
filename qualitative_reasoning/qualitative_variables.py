"""
Qualitative Variables for Qualitative Physics
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np


class VariableType(Enum):
    """Types of qualitative variables"""
    QUANTITY = "quantity"
    DERIVATIVE = "derivative" 
    LANDMARK = "landmark"
    INTERVAL = "interval"
    FLOW_RATE = "flow_rate"
    AMOUNT = "amount"


class QualitativeValue(Enum):
    """Qualitative values for variables"""
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1
    UNKNOWN = "unknown"
    INCREASING = "inc"
    DECREASING = "dec"
    STEADY = "std"
    
    @staticmethod
    def add(val1: 'QualitativeValue', val2: 'QualitativeValue') -> 'QualitativeValue':
        """Add two qualitative values using qualitative arithmetic"""
        if val1 == QualitativeValue.UNKNOWN or val2 == QualitativeValue.UNKNOWN:
            return QualitativeValue.UNKNOWN
        
        # Handle qualitative arithmetic according to qualitative reasoning principles
        if val1 == QualitativeValue.ZERO:
            return val2
        elif val2 == QualitativeValue.ZERO:
            return val1
        elif val1 == val2:
            # Same sign: positive + positive = positive, negative + negative = negative
            return val1
        elif (val1 == QualitativeValue.POSITIVE and val2 == QualitativeValue.NEGATIVE) or \
             (val1 == QualitativeValue.NEGATIVE and val2 == QualitativeValue.POSITIVE):
            # Opposing forces: result is ambiguous/unknown
            return QualitativeValue.UNKNOWN
        else:
            return QualitativeValue.UNKNOWN


class QualitativeVariable:
    """Represents a qualitative variable in the physics simulation"""
    
    def __init__(self, name: str, var_type: VariableType, 
                 value: QualitativeValue = QualitativeValue.UNKNOWN):
        self.name = name
        self.var_type = var_type  # Changed from variable_type to match test expectations
        self.value = value
        self.derivative = QualitativeValue.UNKNOWN
        self.constraints = []
        self.influences = []
        self.landmarks = {}
        
    def set_value(self, value: QualitativeValue):
        """Set the qualitative value"""
        self.value = value
        
    def get_value(self) -> QualitativeValue:
        """Get the current qualitative value"""
        return self.value
        
    def set_derivative(self, derivative: QualitativeValue):
        """Set the derivative (rate of change)"""
        self.derivative = derivative
        
    def get_derivative(self) -> QualitativeValue:
        """Get the derivative (rate of change)"""
        return self.derivative
        
    def add_influence(self, other_var: 'QualitativeVariable', influence_type: QualitativeValue):
        """Add an influence relationship to another variable"""
        self.influences.append((other_var, influence_type))
        
    def get_influences(self) -> List[tuple]:
        """Get all influence relationships"""
        return self.influences
        
    def add_landmark(self, name: str, value: float):
        """Add a landmark value"""
        self.landmarks[name] = value
        
    def get_landmarks(self) -> Dict[str, float]:
        """Get all landmark values"""
        return self.landmarks
        
    def add_constraint(self, constraint):
        """Add a constraint involving this variable"""
        self.constraints.append(constraint)
        
    def __str__(self):
        return f"{self.name}: {self.value}"
        
    def __repr__(self):
        return f"QualitativeVariable('{self.name}', {self.var_type}, {self.value})"