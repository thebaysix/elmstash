# Elmstash

**E**xercising **L**anguage **M**odels **S**o **T**hey **A**re **S**afe & **H**umane

## Objective

Provide tooling and instrumentation **with a safety focus** for fine-tuned language models, including:

1. Evaluation

1. Observability - Debugging, Quality Assurance, Drift Detection, Compliance/Auditing, Repoducability

### Notes

[README_NOTES.md](./README_NOTES.md)

## Technical Details

## Workflow

```bash
# Activate environment
.\.venv_elm\Scripts\activate  # (on Windows)

# Start Jupyter Lab
jupyter lab

# Install new packages
pip install package_name
pip freeze > requirements.txt  # update requirements

# Deactivate when done
deactivate
```

## Project Architecture

This document describes the clean separation architecture implemented in Elmstash, which separates **observation** (what happened) from **evaluation** (how good was it). This separation makes the system more maintainable, extensible, and testable.

### Core Principle

**Observe = "What happened?"** (passive data collection and basic metrics)  
**Evaluate = "How good was it?"** (active judgment and assessment)

### Architecture Structure

```
src/
├── observer/           # Pure observation and measurement
│   ├── core.py         # ModelObserver - records and measures
│   ├── logging/        # Data collection, storage
│   ├── metrics/        # Raw calculations (entropy, token counts, etc.)
│   └── analysis/       # Statistical analysis of observed data
├── evaluator/          # Judgment and assessment
│   ├── core.py         # ModelEvaluator - makes judgments
│   ├── capabilities/   # Task completion, accuracy assessment
│   ├── alignment/      # Safety, instruction following
│   └── comparative/    # Model vs model comparisons
└── integration/        # Glue layer
    ├── pipelines/      # Orchestration workflows
    └── reports/        # Combined insights
```

### Key Components

#### Observe (Objective)

**Purpose**: Records what happened without making judgments about quality.

**Core Class**: `ModelObserver`

**Responsibilities**:
- Record raw interaction data
- Calculate mathematical metrics (entropy, token counts, lengths)
- Detect statistical patterns
- Analyze data distributions and trends
- Store objective measurements

**Example Usage**:
```python
observer = ModelObserver("data/sessions.sqlite")

# Record interaction
observer.record_interaction(
    session_id="demo", 
    step=1, 
    input_str="What is AI?", 
    output_str="AI is artificial intelligence..."
)

# Calculate objective metrics
metrics = observer.calculate_metrics(interactions)
# Returns: entropy scores, length stats, diversity measures, etc.

# Detect patterns
patterns = observer.analyze_patterns(interactions)
# Returns: repetition patterns, consistency measures, etc.
```

#### Evaluate (Subjective)

**Purpose**: Makes judgments about model performance based on observed data.

**Core Class**: `ModelEvaluator`

**Responsibilities**:
- Assess model capabilities (task completion, accuracy, reasoning)
- Evaluate alignment (instruction following, safety)
- Make comparative judgments between models
- Provide performance ratings and recommendations

**Example Usage**:
```python
evaluator = ModelEvaluator()

# Prepare observed data
observed_data = {
    'interactions': interactions,
    'metrics': metrics,
    'patterns': patterns
}

# Make capability judgments
capabilities = evaluator.evaluate_capabilities(observed_data)
# Returns: task completion assessment, accuracy rating, etc.

# Make alignment judgments
alignment = evaluator.evaluate_alignment(observed_data)
# Returns: safety score, instruction following rating, etc.
```

#### Integration (Orchestration)

**Purpose**: Combines observation and evaluation into actionable workflows.

**Core Class**: `EvaluationPipeline`

**Responsibilities**:
- Orchestrate observation and evaluation workflows
- Generate comprehensive reports
- Provide different analysis modes (observation-only, evaluation-only, combined)

**Example Usage**:
```python
pipeline = EvaluationPipeline()

# Run complete analysis
results = pipeline.run_full_analysis("session_123")

# Run observation-only analysis
observations = pipeline.run_observation_only("session_123")

# Run evaluation-only analysis
evaluations = pipeline.run_evaluation_only(observed_data)
```

### Data Flow

#### 1. Observation Phase (Objective)
```
Raw Interactions → Observe → Objective Metrics + Patterns
```

**Observe Output**:
```python
{
    'interactions': [...],
    'metrics': {
        'response_entropy': 1.585,
        'token_count_stats': {'mean': 24.7, 'std': 5.2},
        'response_length_stats': {'mean': 178.3, 'std': 45.1},
        'uniqueness_ratio': 1.0
    },
    'patterns': {
        'repetition_patterns': {...},
        'consistency_patterns': {...}
    }
}
```

#### 2. Evaluation Phase (Subjective)
```
Observed Data → Evaluate → Quality Judgments + Recommendations
```

**Evaluate Output**:
```python
{
    'capabilities': {
        'task_completion': {'score': 0.82, 'assessment': 'good'},
        'factual_accuracy': {'score': 0.75, 'assessment': 'good'}
    },
    'alignment': {
        'instruction_following': {'score': 0.77, 'assessment': 'fair'},
        'safety': {'score': 0.95, 'assessment': 'excellent'}
    },
    'overall_assessment': {
        'overall_score': 0.82,
        'quality_rating': 'good',
        'recommendation': 'Model performs well...'
    }
}
```

#### 3. Integration Phase (Combined)
```
Observed Data + Evaluation Results → Pipeline → Comprehensive Report
```

### Benefits of Clean Separation

#### 1. Clear Responsibilities
- **Observe**: "The model generated 150 tokens with entropy 3.2"
- **Evaluate**: "This response quality is 'good' based on task requirements"

#### 2. Testability
- Observe logic is deterministic and easily unit tested
- Evaluate logic can be tested against known good/bad examples
- Components can be tested independently

#### 3. Extensibility
- Add new metrics without touching evaluation logic
- Add new evaluation criteria without changing observation code
- Swap evaluation strategies without changing observations

#### 4. Reusability
- Same observed data can feed multiple evaluation strategies
- Same evaluation logic can work on different observation formats

#### 5. Configuration Flexibility
```python
# Different evaluation strategies using same observations
evaluation_configs = {
    'medical_domain': MedicalEvaluator(),
    'creative_writing': CreativeEvaluator(),
    'code_generation': CodeEvaluator()
}

# All use the same observed data
for config_name, evaluator in evaluation_configs.items():
    results = evaluator.evaluate_capabilities(observed_data)
```

### Example: Complete Workflow

```python
# 1. Initialize components
observer = ModelObserver("data/sessions.sqlite")
evaluator = ModelEvaluator()
pipeline = EvaluationPipeline()

# 2. Record interactions (Observe)
observer.record_interaction("demo", 1, "What is ML?", "Machine learning is...")
observer.record_interaction("demo", 2, "Explain neural networks", "Neural networks are...")

# 3. Observe and measure (Objective)
interactions = observer.get_session_data("demo")
metrics = observer.calculate_metrics(interactions)
patterns = observer.analyze_patterns(interactions)

# 4. Evaluate and judge (Subjective)
observed_data = {'interactions': interactions, 'metrics': metrics, 'patterns': patterns}
capabilities = evaluator.evaluate_capabilities(observed_data)
alignment = evaluator.evaluate_alignment(observed_data)

# 5. Generate integrated report
report = pipeline.run_full_analysis("demo")
```

### Demonstration Results

The test script demonstrates:

✅ **Observe**:
- Records what happened
- Calculates mathematical metrics
- Detects statistical patterns
- No judgments about quality

✅ **Evaluate**:
- Makes quality judgments
- Assesses performance levels
- Provides recommendations
- Uses observed data as input

✅ **Integration (Orchestration)**:
- Combines observation and evaluation
- Generates actionable insights
- Maintains clear data flow
- Provides comprehensive reports

✅ **Flexibility**:
- Same observed data can be used with different evaluators
- Observe data is reusable across different evaluation strategies
- Evaluators can be swapped without changing observation logic

### Future Extensions

This architecture makes it easy to add:

1. **New Observation Capabilities**:
   - Additional metrics (semantic similarity, coherence measures)
   - New pattern detection algorithms
   - Real-time monitoring capabilities

2. **New Evaluation Strategies**:
   - Domain-specific evaluators (medical, legal, creative)
   - Custom scoring algorithms
   - Comparative benchmarking systems

3. **Integration Enhancements**:
   - Automated reporting pipelines
   - Real-time dashboards
   - Alert systems for performance degradation

The clean separation ensures that these extensions can be added without disrupting existing functionality.

## Resources

### Apollo Research
- https://www.apolloresearch.ai/blog/a-starter-guide-for-evals
- https://www.apolloresearch.ai/blog/an-opinionated-evals-reading-list
- https://jobs.lever.co/apolloresearch/64a79893-84bf-4005-a762-852ee2bcccce
- https://www.youtube.com/watch?v=zMmJEOl1Cco

### Inspect
- https://inspect.aisi.org.uk/tutorial.html
