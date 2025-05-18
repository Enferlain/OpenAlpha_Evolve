import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Attempt to load the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback for development if .env is not set or key is not found,
# but ensure this is handled securely in production.
if not GEMINI_API_KEY:
    # --- IMPORTANT ---
    # Directly embedding keys is a security risk.
    # This is a placeholder for local development ONLY.
    # In a real deployment, use environment variables, secrets management, or other secure methods.
    # For local testing without a .env file, you can temporarily set it like:
    # GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
    print("Warning: GEMINI_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
    GEMINI_API_KEY = "YOUR_API_KEY_FROM_DOTENV_WAS_NOT_FOUND_PLEASE_SET_IT_UP" # Obvious placeholder

# LLM Model Configuration
GEMINI_PRO_MODEL_NAME = "gemini-2.5-flash-preview-04-17" #"gemini-2.0-flash-lite" # Using a more capable model
GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash-preview-04-17" #"gemini-2.0-flash-lite" # Default model for speed
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash-preview-04-17" #"gemini-2.0-flash-lite" # Model for evaluation tasks

# Evolutionary Parameters (examples)
POPULATION_SIZE = 10  # Number of individuals in each generation
GENERATIONS = 10       # Number of generations to run the evolution
ELITISM_COUNT = 1     # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.7   # Probability of mutating an individual
CROSSOVER_RATE = 0.2  # Probability of crossing over two parents (if crossover is implemented)

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = 800  # Max time for a program to run during evaluation

# Database settings (using a simple in-memory store for now)
DATABASE_TYPE = "in_memory" # or "sqlite", "postgresql" in the future
# DATABASE_PATH = "program_database.json" # Before
DATABASE_PATH = "alpha_evolve_programs.db" # After, to match SQLite agent default or desired .db name

# Logging Parameters
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpha_evolve.log"

# API Retry Parameters
API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10 # Initial delay, will be exponential

# Placeholder for RL Fine-Tuning (if implemented)
RL_TRAINING_INTERVAL_GENERATIONS = 50 # Fine-tune RL model every N generations
RL_MODEL_PATH = "rl_finetuner_model.pth"

METRIC_OPTIMIZATION_DIRECTION = {
    "correctness": True, "pylint_score": True, "maintainability_index": True,
    "passed_tests": True, "runtime_ms": False, "cyclomatic_complexity_avg": False,
}
DEFAULT_METRIC_VALUE = {
    "correctness": 0.0, "pylint_score": -1.0, "maintainability_index": -1.0,
    "passed_tests": 0.0, "runtime_ms": float('inf'), "cyclomatic_complexity_avg": float('inf'),
}

# Monitoring (if implemented)
MONITORING_DASHBOARD_URL = "http://localhost:8080" # Example

# --- Helper function to get a specific setting ---
def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    # Prioritize environment variables for some settings if needed
    # For example: return os.getenv(key, globals().get(key, default))
    return globals().get(key, default)

# Example of how to get a model, perhaps with fallback logic (not strictly necessary with current direct assignments)
def get_llm_model(model_type="pro"):
    if model_type == "pro":
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME # Default fallback

# Add other global settings here 
