# test_inspect.py
from inspect_ai import Task, eval
from inspect_ai.dataset import example_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact

# Create a simple task
@task
def simple_eval():
    return Task(
        dataset=example_dataset("math_word_problems"),
        plan=[generate()],
        scorer=model_graded_fact()
    )

# Run evaluation (when ready)
# eval(simple_eval(), model="gpt-3.5-turbo")