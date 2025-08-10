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

## Project Structure

### Eval

TBD...

### Observer

```bash
observer/
├── sandbox/
│   └── simple_sandbox.py  # Micro closed/open scenario
├── agent/
│   └── model_interface.py  # Wrapper for OpenAI/Anthropic
├── logging/
│   ├── db.py               # SQLite schema + logging functions
│   └── logger.py           # Unified logging API
├── metrics/
│   ├── entropy.py          # Input entropy
│   ├── info_gain.py        # Dirichlet-based info gain
│   └── empowerment.py      # Action-input mutual info
├── notebooks/
│   └── prototype.ipynb     # Where you do dev & demo
├── visualization/
│   └── plots.py            # Matplotlib or Plotly visualizers
├── data/
│   └── sessions.sqlite     # Logs stored here
```

## Resources

### Apollo Research
- https://www.apolloresearch.ai/blog/a-starter-guide-for-evals
- https://www.apolloresearch.ai/blog/an-opinionated-evals-reading-list
- https://jobs.lever.co/apolloresearch/64a79893-84bf-4005-a762-852ee2bcccce

### Inspect
- https://inspect.aisi.org.uk/tutorial.html
