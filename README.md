# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/qualitative-reasoning/workflows/CI/badge.svg)](https://github.com/benedictchen/qualitative-reasoning/actions)
[![PyPI version](https://badge.fury.io/py/qualitative-reasoning.svg)](https://badge.fury.io/py/qualitative-reasoning)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Qualitative Reasoning

🔬 Qualitative physics simulation and causal reasoning systems

**Forbus, K. D. (1984)** - "Qualitative process theory"  
**de Kleer, J., & Brown, J. S. (1984)** - "A qualitative physics based on confluences"

## 📦 Installation

```bash
pip install qualitative-reasoning
```

## 🚀 Quick Start

### Basic Qualitative Physics Simulation
```python
from qualitative_reasoning import QualitativePhysicsEngine
import numpy as np

# Create physics simulation engine
engine = QualitativePhysicsEngine(
    simulation_method='envisionment',
    reasoning_mode='causal',
    temporal_logic=True
)

# Define a simple physical system (water flow)
system = {
    'containers': ['tank_a', 'tank_b'],
    'connections': [('tank_a', 'tank_b', 'pipe')],
    'initial_state': {
        'tank_a': {'water_level': 'high', 'pressure': 'high'},
        'tank_b': {'water_level': 'low', 'pressure': 'low'}
    }
}

# Run qualitative simulation
simulation = engine.simulate(system, time_steps=10)
print("Simulation states:", simulation.get_state_sequence())

# Analyze causal relationships
causal_graph = engine.infer_causality(simulation)
print("Causal relationships:", causal_graph.get_edges())
```

### Confluence-based Reasoning
```python
from qualitative_reasoning import ConfluenceEngine

# Create confluence reasoning system
confluence = ConfluenceEngine(
    constraint_propagation=True,
    consistency_checking=True
)

# Define confluences for electrical circuit
confluences = [
    ('voltage', 'current', 'resistance', 'ohms_law'),
    ('current', 'time', 'charge', 'integration'),
    ('power', 'voltage', 'current', 'power_law')
]

confluence.add_confluences(confluences)

# Set initial conditions
confluence.set_variable('voltage', 'increasing')
confluence.set_variable('resistance', 'constant')

# Propagate constraints
result = confluence.propagate()
print("Inferred values:")
for var, value in result.items():
    print(f"  {var}: {value}")
```

### Envisionment and State Transitions
```python
from qualitative_reasoning import Envisionment

# Create state space exploration
envisionment = Envisionment(
    state_representation='quantity_space',
    transition_rules='process_based',
    reachability_analysis=True
)

# Define quantity spaces
quantity_spaces = {
    'temperature': ['cold', 'warm', 'hot'],
    'pressure': ['low', 'medium', 'high'], 
    'volume': ['small', 'medium', 'large']
}

envisionment.define_quantity_spaces(quantity_spaces)

# Define processes and transitions
processes = [
    {
        'name': 'heating',
        'conditions': {'temperature': 'cold'},
        'effects': {'temperature': 'increase', 'pressure': 'increase'}
    },
    {
        'name': 'expansion', 
        'conditions': {'pressure': 'high'},
        'effects': {'volume': 'increase', 'pressure': 'decrease'}
    }
]

envisionment.add_processes(processes)

# Generate state transition graph
state_graph = envisionment.generate_envisionment()
print(f"Total states: {len(state_graph.nodes)}")
print(f"Possible transitions: {len(state_graph.edges)}")
```

## 🧠 Advanced Causal Reasoning

### Assumption-based Truth Maintenance
```python
from qualitative_reasoning import CausalReasoning

# Create causal reasoning system
causal = CausalReasoning(
    truth_maintenance='atms',  # Assumption-based TMS
    dependency_tracking=True,
    contradiction_handling='backtrack'
)

# Model causal relationships in mechanical system
causal_model = {
    'assumptions': [
        ('gear_a_rotating', 'clockwise'),
        ('gear_connection', 'engaged')
    ],
    'rules': [
        ('gear_a_rotating ∧ gear_connection → gear_b_rotating'),
        ('gear_b_rotating ∧ load_present → torque_required'),
        ('torque_required > motor_capacity → system_failure')
    ]
}

causal.load_model(causal_model)

# Reason about system behavior
result = causal.reason()
print("Derived conclusions:", result.conclusions)
print("Supporting assumptions:", result.assumptions)

# Handle contradictions
if result.contradictions:
    print("Contradictions found:", result.contradictions)
    resolution = causal.resolve_contradictions()
    print("Resolution:", resolution)
```

### Constraint Propagation
```python
from qualitative_reasoning import ConstraintPropagation

# Constraint-based qualitative reasoning
constraints = ConstraintPropagation(
    propagation_algorithm='ac3',
    consistency_level='arc_consistency'
)

# Define constraint network for spatial reasoning
constraint_network = {
    'variables': ['object_a', 'object_b', 'object_c'],
    'domains': {
        'object_a': ['left', 'center', 'right'],
        'object_b': ['left', 'center', 'right'],
        'object_c': ['left', 'center', 'right']
    },
    'constraints': [
        ('object_a', 'object_b', 'left_of'),
        ('object_b', 'object_c', 'adjacent'),
        ('object_a', 'object_c', 'not_same')
    ]
}

constraints.load_network(constraint_network)

# Propagate constraints and find solutions
solutions = constraints.solve()
print("Valid configurations:")
for solution in solutions:
    print(f"  {solution}")
```

## 🔬 Key Algorithmic Features

### Qualitative Process Theory
- **Process Modeling**: Representation of continuous processes and their effects
- **Quantity Spaces**: Discrete abstractions of continuous variables
- **State Transitions**: Qualitative differential equations and behavior prediction
- **Temporal Reasoning**: Event ordering and temporal constraint satisfaction

### Confluence-based Physics
- **Confluence Modeling**: Local constraint propagation in physical systems
- **Constraint Networks**: Global consistency maintenance across variables
- **Ambiguity Resolution**: Handling multiple possible interpretations
- **Incremental Reasoning**: Efficient updates to changing systems

### Envisionment Generation
- **State Space Exploration**: Systematic generation of possible system states
- **Reachability Analysis**: Determining possible future states from current conditions
- **Behavior Trees**: Hierarchical organization of system behaviors
- **Abstraction Levels**: Multiple granularities of system description

## 📊 Implementation Highlights

- **Research Accuracy**: Faithful implementation of foundational QR algorithms
- **Modular Architecture**: Clean separation of reasoning components
- **Educational Value**: Clear implementation for learning qualitative reasoning
- **Performance Optimized**: Efficient constraint propagation and search
- **Extensible Framework**: Easy to add new reasoning methods

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**