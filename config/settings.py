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
MODEL_GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite" # Alias for gemini-2.0-flash-lite-001
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"           # Alias for gemini-2.0-flash-001
MODEL_GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-04-17"
MODEL_GEMINI_2_5_PRO_PREVIEW = "gemini-2.5-pro-preview-05-06" # Corrected name if it's 05-06

# --- Select CURRENTLY Active Model for Code Generation ---
# This is the model that CodeProducer will use by default
# and whose RPM limit EvolveFlow will primarily respect for its global semaphore.
# GENERATION_MODEL_NAME = MODEL_GEMINI_2_5_FLASH_PREVIEW # Default to this one for now
GENERATION_MODEL_NAME = MODEL_GEMINI_2_0_FLASH_LITE # Or this one for more RPM

# --- Model-Specific Free Tier RPM Limits ---
# (Requests Per Minute for the free tier)
MODEL_FREE_TIER_RPM = {
    MODEL_GEMINI_2_0_FLASH_LITE: 30,
    MODEL_GEMINI_2_0_FLASH: 15,
    MODEL_GEMINI_2_5_FLASH_PREVIEW: 10,
    MODEL_GEMINI_2_5_PRO_PREVIEW: 5,
    # Add any other models and their RPMs here if needed
    # Fallback RPM if model not listed (be conservative)
    "default": 5
}

# --- Other LLM Model Configuration ---
# GEMINI_FLASH_MODEL_NAME was used before, maybe consolidate or use for a different purpose?
# For now, let's assume GEMINI_PRO_MODEL_NAME is the main one for generation.
# EVALUATION_LLM_MODEL_NAME could be different if needed for specific eval tasks by an LLM.
EVALUATION_MODEL_NAME = MODEL_GEMINI_2_5_FLASH_PREVIEW # Example

# Evolutionary Parameters (examples)
POPULATION_SIZE = 1  # Number of individuals in each generation
GENERATIONS = 10      # Number of generations to run the evolution
ELITISM_COUNT = 1     # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.7   # Probability of mutating an individual
CROSSOVER_RATE = 0.2  # Probability of crossing over two parents

# Model settings
TEMPERATURE_INITIAL_GEN = 1.0
TEMPERATURE_CROSSOVER = 1.0
TEMPERATURE_MUTATION_DIFF = 0.75
TEMPERATURE_FALLBACK_FULL = 0.70

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = 800  # Max time for a program to run during evaluation

# Database settings (using a simple in-memory store for now)
DATABASE_TYPE = "in_memory" # or "sqlite", "postgresql" in the future
# DATABASE_PATH = "program_database.json" # Before
DATABASE_PATH = "alpha_evolve_programs.db" # After, to match SQLite agent default or desired .db name

# Logging Parameters
LOG_LEVEL = "DEBUG" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpha_evolve.log"

# API Retry Parameters
API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10 # Initial delay, will be exponential

# Placeholder for RL Fine-Tuning (if implemented)
RL_TRAINING_INTERVAL_GENERATIONS = 50 # Fine-tune RL model every N generations
RL_MODEL_PATH = "rl_finetuner_model.pth"

METRIC_OPTIMIZATION_DIRECTION = {
    "correctness": True,  # Higher is better
    "ai_review_score": True,
    "ruff_violations": False, # Lower is better!
    "maintainability_index": True,
    "passed_tests": True,
    "runtime_ms": False,  # Lower is better
    "cyclomatic_complexity_avg": False, # Lower is better
}
DEFAULT_METRIC_VALUE = {
    "correctness": 0.0,
    "ruff_violations": float('inf'), # Default to very bad
    "maintainability_index": -1.0,
    "passed_tests": 0.0,
    "runtime_ms": float('inf'),
    "cyclomatic_complexity_avg": float('inf'),
}

# Monitoring (if implemented)
MONITORING_DASHBOARD_URL = "http://localhost:8080" # Example

# --- Helper function to get a specific setting ---
def get_setting(key, default=None):
    """
    Retrieves a setting value from this module's global scope.
    """
    return globals().get(key, default)

# --- Updated/Simplified get_llm_model function ---
def get_generation_model() -> str:
    """
    Returns the currently configured primary model name for code generation.
    """
    return GENERATION_MODEL_NAME

def get_evaluation_model() -> str:
    """
    Returns the currently configured model name for LLM-based evaluation tasks.
    """
    return EVALUATION_MODEL_NAME

# You can add more specific getters if needed, e.g.,
# def get_model_rpm(model_name: str) -> int:
#     return MODEL_FREE_TIER_RPM.get(model_name, MODEL_FREE_TIER_RPM["default"])

# Add other global settings here