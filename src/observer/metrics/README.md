# Comprehensive Model Evaluation Metrics

This module provides a comprehensive suite of metrics for evaluating language model performance across multiple dimensions: capabilities, alignment, and information-theoretic measures.

## Overview

The metrics system is organized into three implementation phases:

- **Phase 1**: Core entropy metrics, basic capabilities, and basic alignment
- **Phase 2**: Advanced capabilities and comprehensive alignment measures  
- **Phase 3**: Information-theoretic measures for deep model analysis

## Quick Start

```python
from observer.metrics.evaluation import MetricsEvaluator

# Initialize evaluator (Phase 2 includes Phase 1 metrics)
evaluator = MetricsEvaluator(phase=2)

# Evaluate a single interaction
results = evaluator.evaluate_single_interaction(
    input_prompt="Explain photosynthesis",
    output_response="Photosynthesis is the process...",
    session_id="session_001",
    step=1
)

print(f"Task completion: {results['task_completion']:.3f}")
print(f"Safety score: {results['safety']:.3f}")
```

## Metrics Categories

### Core Entropy Metrics

| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| **Character Entropy** | Linguistic diversity within a single response | Text complexity, vocabulary richness |
| **Response Entropy** | Consistency vs diversity across multiple responses | Model reliability, mode collapse detection |
| **Input Entropy** | Diversity of test cases used | Evaluation coverage assessment |

```python
from observer.metrics.entropy import calc_character_entropy, calc_response_entropy

# Single response diversity
entropy = calc_character_entropy("Your response text here")

# Multiple response consistency
responses = ["Response 1", "Response 2", "Response 3"]
consistency = calc_response_entropy(responses)
```

### Capability Metrics

| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| **Task Completion** | Whether the model solved the requested problem | Core capability assessment |
| **Factual Accuracy** | Correctness of stated facts | Knowledge verification |
| **Reasoning Quality** | Logical coherence and validity | Problem-solving assessment |

```python
from observer.metrics.capabilities import task_completion_score, factual_accuracy_score

# Basic capability check
score = task_completion_score(prompt, response)

# Fact verification
ground_truth = {'capital_france': 'Paris'}
accuracy = factual_accuracy_score(response, ground_truth)
```

### Alignment Metrics

| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| **Instruction Following** | Adherence to specific instructions | Controllability assessment |
| **Helpfulness** | Practical utility of responses | User satisfaction proxy |
| **Safety Score** | Absence of harmful content | Harm prevention verification |

```python
from observer.metrics.alignment import instruction_following_score, safety_score

# Instruction adherence
following = instruction_following_score(prompt, response)

# Safety check
safety = safety_score(response)
if safety < 0.8:
    print("⚠️ Safety concern detected")
```

### Information-Theoretic Metrics (Phase 3)

| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| **Information Gain** | Learning from interactions | Knowledge acquisition efficiency |
| **Empowerment** | Ability to influence outcomes | Agency and effectiveness |
| **Uncertainty Calibration** | Confidence vs accuracy alignment | Honesty assessment |

```python
from observer.metrics.information_theory import information_gain, empowerment

# Learning measurement
gain = information_gain(observations_before, observations_after)

# Agency assessment  
emp = empowerment(model_actions, resulting_outcomes)
```

## Warning Signs Detection

The system automatically detects concerning patterns:

### Mode Collapse
```python
# Very low response entropy indicates repetitive outputs
if calc_response_entropy(responses) < 1.0:
    print("⚠️ Possible mode collapse")
```

### Poor Reliability
```python
# High entropy on factual questions indicates inconsistency
factual_responses = ["Paris", "London", "Berlin"]  # for "Capital of France?"
if calc_response_entropy(factual_responses) > 1.5:
    print("⚠️ Poor factual reliability")
```

### Safety Violations
```python
if safety_score(response) < 0.8:
    print("⚠️ Safety violation detected")
```

## Comprehensive Evaluation

Use the `MetricsEvaluator` class for complete assessment:

```python
evaluator = MetricsEvaluator(phase=2)

# Single interaction
results = evaluator.evaluate_single_interaction(
    input_prompt=prompt,
    output_response=response,
    session_id="test_session",
    step=1,
    ground_truth={'expected_facts': ['fact1', 'fact2']}
)

# Full session analysis
session_results = evaluator.evaluate_session("test_session")

# Multi-session summary
summary = evaluator.get_evaluation_summary(
    session_ids=["session1", "session2"],
    metric_categories=["capabilities", "alignment"]
)
```

## Visualization

The system includes comprehensive plotting capabilities:

```python
from observer.metrics.plots import (
    plot_metrics_dashboard,
    plot_performance_radar,
    plot_safety_analysis
)

# Comprehensive dashboard
plot_metrics_dashboard(session_id="test_session")

# Radar chart comparison
plot_performance_radar("session1", comparison_session="session2")

# Safety-focused analysis
plot_safety_analysis(threshold=0.8)
```

## Implementation Phases

### Phase 1 (Immediate Implementation)
- ✅ Character/Response Entropy
- ✅ Task Completion Score
- ✅ Instruction Following Score

### Phase 2 (Advanced Features)
- ✅ Input Entropy
- ✅ Factual Accuracy Score
- ✅ Reasoning Quality Score
- ✅ Helpfulness Score
- ✅ Safety Score

### Phase 3 (Research-Level)
- ✅ Information Gain
- ✅ Empowerment
- ✅ Uncertainty Calibration

## Database Integration

All metrics are automatically stored in SQLite for historical analysis:

```sql
-- Evaluation results table structure
CREATE TABLE evaluation_results (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    step INTEGER,
    timestamp TEXT,
    character_entropy REAL,
    task_completion REAL,
    instruction_following REAL,
    factual_accuracy REAL,
    reasoning_quality REAL,
    helpfulness REAL,
    safety REAL,
    information_gain REAL,
    empowerment REAL,
    metadata TEXT
);
```

## Metric Interpretation

### Score Ranges
- **0.0 - 0.3**: Poor performance, needs attention
- **0.3 - 0.7**: Moderate performance, room for improvement  
- **0.7 - 1.0**: Good to excellent performance

### Entropy Interpretation
- **Low entropy (< 2.0 bits)**: High consistency, possible over-constraint
- **Medium entropy (2.0 - 4.0 bits)**: Balanced diversity
- **High entropy (> 4.0 bits)**: High diversity, possible inconsistency

### Safety Thresholds
- **> 0.9**: Very safe
- **0.8 - 0.9**: Generally safe
- **< 0.8**: Potential safety concerns

## Best Practices

1. **Use appropriate phases**: Start with Phase 1, add Phase 2 for production systems
2. **Monitor trends**: Track metrics over time, not just individual scores
3. **Combine metrics**: Use multiple metrics together for holistic assessment
4. **Set thresholds**: Define acceptable ranges for your specific use case
5. **Regular evaluation**: Implement continuous monitoring in production

## Dependencies

Core functionality works with Python standard library only. Optional features require:

- `matplotlib`, `seaborn`: For visualization
- `pandas`: For advanced data analysis
- `numpy`: For numerical computations

## Testing

Run the test suite to verify functionality:

```bash
python test_metrics.py
```

Run the demonstration:

```bash
python demo_metrics.py
```

## Contributing

When adding new metrics:

1. Follow the existing pattern in the appropriate category module
2. Add comprehensive docstrings with examples
3. Include the metric in the `MetricsEvaluator` class
4. Add visualization support in `plots.py`
5. Update this README with usage examples

## License

This metrics system is part of the Elmstash project and follows the same licensing terms.