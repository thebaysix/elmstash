# simple_sandbox.py

def closed_prompt():
    return {
        "prompt": "Patient has a headache. What do you do?",
        "options": [
            "Give ibuprofen",
            "Ask about vision",
            "Refer to specialist"
        ]
    }

def open_prompt():
    return "A patient arrives complaining of a headache. What do you do?"
