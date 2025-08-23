---
description: Repository Information Overview
alwaysApply: true
---

# Elmstash Information

## Summary
Elmstash (Exercising Language Models So They Are Safe & Humane) is a Python project focused on providing tooling and instrumentation for fine-tuned language models with a safety focus. It includes components for evaluation, observability, debugging, quality assurance, drift detection, compliance/auditing, and reproducibility.

## Structure
The repository is organized into several main components:
- **src/observer**: Provides monitoring and logging capabilities for LLM interactions
- **src/evaluator**: Makes judgments about model performance based on observed data
- **src/eval**: Implements evaluation metrics using inspect-ai framework
- **src/integration**: Orchestrates observation and evaluation workflows
- **notebooks**: Contains Jupyter notebooks for development and demonstrations
- **data**: Stores SQLite databases for logging model interactions

## Language & Runtime
**Language**: Python
**Version**: 3.12.3
**Package Manager**: pip
**Environment**: Virtual environment (.venv_elm)

## Dependencies
**Main Dependencies**:
- jupyterlab (≥4.0.0) - Interactive development environment
- inspect-ai (≥0.3.0) - Evaluation framework
- openai (≥1.0.0) - OpenAI API client
- anthropic (≥0.7.0) - Anthropic API client
- pandas (≥2.0.0) - Data manipulation
- matplotlib (≥3.7.0), plotly (≥5.15.0), seaborn (≥0.12.0) - Visualization
- numpy (≥1.24.0) - Numerical computing
- python-dotenv (≥1.0.0) - Environment variable management

## Build & Installation
```bash
# Create and activate virtual environment
python -m venv .venv_elm
.\.venv_elm\Scripts\activate  # (on Windows)

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab for development
jupyter lab
```

## Projects

### Observer Component
**Configuration File**: src/observer/core.py

#### Main Features
- **Model Interface**: Wrapper for OpenAI/Anthropic APIs (src/observer/agent/model_interface.py)
- **Logging System**: SQLite-based logging for model interactions (src/observer/logging/db.py)
- **Metrics**: Implements entropy, information gain, and empowerment metrics (src/observer/metrics/)
- **Sandbox**: Provides test environments for model evaluation (src/observer/sandbox/)

#### Usage
```python
# Initialize observer
from observer.core import ModelObserver
observer = ModelObserver("data/sessions.sqlite")

# Record interaction
observer.record_interaction(
    session_id="demo", 
    step=1, 
    input_str="What is AI?", 
    output_str="AI is artificial intelligence..."
)

# Calculate metrics
metrics = observer.calculate_metrics(interactions)
patterns = observer.analyze_patterns(interactions)
```

### Evaluator Component
**Configuration File**: src/evaluator/core.py

#### Main Features
- **Capability Evaluation**: Assesses task completion, accuracy, reasoning (src/evaluator/capabilities/)
- **Alignment Evaluation**: Evaluates instruction following, safety (src/evaluator/alignment/)
- **Comparative Evaluation**: Compares performance between models (src/evaluator/comparative/)

#### Usage
```python
# Initialize evaluator
from evaluator.core import ModelEvaluator
evaluator = ModelEvaluator()

# Prepare observed data
observed_data = {
    'interactions': interactions,
    'metrics': metrics,
    'patterns': patterns
}

# Evaluate model
capabilities = evaluator.evaluate_capabilities(observed_data)
alignment = evaluator.evaluate_alignment(observed_data)
results = evaluator.evaluate_comprehensive(observed_data)
```

### Eval Component
**Configuration File**: src/eval/test_inspect.py

#### Main Features
- Utilizes inspect-ai for model evaluation
- Implements various evaluation metrics:
  - Character/Token Entropy
  - Response Entropy
  - Input Entropy
  - Information Gain
  - Empowerment
  - Task Completion
  - Factual Accuracy
  - Reasoning Quality
  - Instruction Following
  - Safety Score

#### Usage
```python
from inspect_ai import Task, eval
from inspect_ai.dataset import example_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact

# Create evaluation task
@task
def simple_eval():
    return Task(
        dataset=example_dataset("math_word_problems"),
        plan=[generate()],
        scorer=model_graded_fact()
    )

# Run evaluation
eval(simple_eval(), model="gpt-3.5-turbo")
```

### Integration Component
**Configuration File**: src/integration/__init__.py

#### Main Features
- **Pipelines**: Orchestrates observation and evaluation workflows (src/integration/pipelines/)
- **Reports**: Generates comprehensive reports from analysis results (src/integration/reports/)

#### Usage
```python
# Initialize pipeline
from integration.pipelines import EvaluationPipeline
pipeline = EvaluationPipeline()

# Run complete analysis
results = pipeline.run_full_analysis("session_123")

# Generate report
from integration.reports import ReportGenerator
report = ReportGenerator().generate_report(results)
```

## Testing
**Framework**: inspect-ai
**Test Location**: src/eval/
**Run Command**:
```bash
# Tests are primarily run through Jupyter notebooks
jupyter lab notebooks/prototype.ipynb
```