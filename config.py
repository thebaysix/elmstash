import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Data paths
DATA_DIR = 'data'
RESULTS_DIR = 'results'
NOTEBOOKS_DIR = 'notebooks'

# Visualization settings
PLOTLY_TEMPLATE = 'plotly_white'
MATPLOTLIB_STYLE = 'seaborn-v0_8'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
