---
description: Repository Information Overview
alwaysApply: true
---

# Elmstash Information

## Summary
Elmstash (Exercising Language Models So They Are Safe & Humane) is a Python project focused on providing tooling and instrumentation for fine-tuned language models with a safety focus. It includes components for evaluation, observability, debugging, quality assurance, drift detection, compliance/auditing, and reproducibility.

## Structure
The repository is organized into two main components:
- **src/observer**: Provides monitoring and logging capabilities for LLM interactions
- **src/eval**: Implements evaluation metrics and testing frameworks for LLMs
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
**Configuration File**: src/observer/logging/logger.py

#### Main Features
- **Model Interface**: Wrapper for OpenAI/Anthropic APIs (src/observer/agent/model_interface.py)
- **Logging System**: SQLite-based logging for model interactions (src/observer/logging/db.py)
- **Metrics**: Implements entropy, information gain, and empowerment metrics (src/observer/metrics/)
- **Sandbox**: Provides test environments for model evaluation (src/observer/sandbox/)

#### Usage
```python
# Initialize database
from observer.logging.logger import init_db, log_interaction
conn = init_db()

# Query model
from observer.agent.model_interface import query_model
response = query_model("Your prompt here", model="gpt-4o-mini")

# Calculate metrics
from observer.metrics.entropy import calc_entropy
entropy = calc_entropy(response)

# Log interaction
log_interaction(conn, session_id, step, input_str, action, output_str, metadata)
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

## Testing
**Framework**: inspect-ai
**Test Location**: src/eval/
**Run Command**:
```bash
# Tests are primarily run through Jupyter notebooks
jupyter lab notebooks/prototype.ipynb
```