import asyncio
import logging
import sys
import time
import argparse  # For command-line arguments
import os  # For path operations
import yaml  # For loading YAML files (pip install PyYAML)

from flow.controller import TaskManagerAgent
from core.interfaces import TaskDefinition  # Assuming Program is also imported if needed by TaskDefinition
from config import settings

# --- Logging Setup ---
# Determine log level from settings, default to INFO
log_level_str = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)

# Prepare handlers
handlers = [logging.StreamHandler(sys.stdout)]
log_file_path = getattr(settings, 'LOG_FILE', None)
if log_file_path:
    try:
        # Ensure the directory for the log file exists if it's specified with a path
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, mode='a'))  # Append mode
        print(f"Logging to file: {log_file_path}")  # Useful bootstrap message
    except Exception as e_log:
        print(f"Warning: Could not set up file logging for {log_file_path}: {e_log}", file=sys.stderr)

logging.basicConfig(
    level=numeric_log_level,
    format="%(asctime)s - %(name)s [%(levelname)s] - %(message)s",  # Slightly more detailed format
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)
logger = logging.getLogger(__name__)  # Logger for main.py


# --- Helper to convert YAML's .inf/.nan to Python's float values ---
# PyYAML does this automatically when loading if they are unquoted,
# but if they are strings like "float('inf')", this explicit conversion is not needed.
# The .inf YAML standard is preferred. TaskDefinition's input_output_examples
# will receive actual float('inf') values if YAML has .inf

async def run_alpha_evolve(cli_args):
    """
    Initializes and runs the OpenAlpha_Evolve Task Manager based on a task configuration file.
    """
    logger.info("--- OpenAlpha_Evolve Starting ---")
    if cli_args.generations is not None:  # Check if CLI arg was passed
        logger.info(f"Overriding settings.GENERATIONS from {settings.GENERATIONS} to {cli_args.generations} via CLI.")
        settings.GENERATIONS = cli_args.generations
    if cli_args.population_size is not None:  # Check if CLI arg was passed
        logger.info(
            f"Overriding settings.POPULATION_SIZE from {settings.POPULATION_SIZE} to {cli_args.population_size} via CLI.")
        settings.POPULATION_SIZE = cli_args.population_size

    logger.info(f"Using effective settings: Population={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")

    # 1. Load Task Configuration from YAML file
    task_config_dict = {}
    if not os.path.exists(cli_args.task_config_file):
        logger.error(f"Task configuration file not found: {cli_args.task_config_file}")
        return

    logger.info(f"Loading task configuration from: {cli_args.task_config_file}")
    try:
        with open(cli_args.task_config_file, 'r', encoding='utf-8') as f:
            task_config_dict = yaml.safe_load(f)
        if not isinstance(task_config_dict, dict):
            logger.error(f"Task configuration file {cli_args.task_config_file} did not parse into a dictionary.")
            return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML from {cli_args.task_config_file}: {e}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Failed to load task_config_file {cli_args.task_config_file}: {e}", exc_info=True)
        return

    # 2. Handle `initial_seed_code_path` if present in the YAML
    #    This allows specifying seed code in a separate file for clarity.
    if "initial_seed_code_path" in task_config_dict:
        seed_code_path_str = task_config_dict.pop("initial_seed_code_path")  # Remove and process
        if seed_code_path_str and isinstance(seed_code_path_str, str):
            # Make path relative to the YAML file's directory or an absolute path
            if not os.path.isabs(seed_code_path_str):
                yaml_dir = os.path.dirname(cli_args.task_config_file)
                resolved_seed_path = os.path.join(yaml_dir, seed_code_path_str)
            else:
                resolved_seed_path = seed_code_path_str

            if os.path.exists(resolved_seed_path):
                try:
                    with open(resolved_seed_path, 'r', encoding='utf-8') as sf:
                        task_config_dict["initial_seed_code"] = sf.read()
                    logger.info(f"Successfully loaded initial_seed_code from: {resolved_seed_path}")
                except Exception as e_seed:
                    logger.error(f"Failed to read initial_seed_code from {resolved_seed_path}: {e_seed}", exc_info=True)
                    # Decide if this is fatal. For now, we'll let TaskDefinition handle missing seed if it's crucial.
                    task_config_dict["initial_seed_code"] = None
            else:
                logger.warning(f"initial_seed_code_path specified ({resolved_seed_path}), but file not found.")
                task_config_dict["initial_seed_code"] = None  # Ensure it's None if path is bad
        elif seed_code_path_str:  # If it's not None but also not a string (e.g. bool, number)
            logger.warning(f"initial_seed_code_path in YAML was not a valid path string: {seed_code_path_str}")

    # 3. Create TaskDefinition object
    #    The `TaskDefinition` dataclass will raise TypeError if required fields are missing
    #    or if types are incorrect, which is good for validation.
    try:
        current_task = TaskDefinition(**task_config_dict)
    except TypeError as e:
        logger.error(
            f"Error creating TaskDefinition from config '{cli_args.task_config_file}'. Missing or incorrect field? Details: {e}",
            exc_info=True)
        logger.error(f"Effective dictionary passed to TaskDefinition: {task_config_dict}")
        return
    except Exception as e_task_def:  # Catch any other unexpected error during TaskDefinition creation
        logger.error(f"Unexpected error creating TaskDefinition: {e_task_def}", exc_info=True)
        return

    logger.info(
        f"Successfully created TaskDefinition for task: '{current_task.id}' (Mode: {current_task.improvement_mode})")

    # --- CORRECTED LOGGING for initial_seed_code_or_ideas ---
    if current_task.initial_seed_code_or_ideas:
        logger.info(
            f"  Initial seed code or ideas will be used (length: {len(current_task.initial_seed_code_or_ideas)} characters).")
    # --- END CORRECTION ---

    if current_task.primary_focus_metrics:
        logger.info(f"  Primary focus metrics: {current_task.primary_focus_metrics}")
    if current_task.specific_improvement_directives:
        logger.info(f"  Specific improvement directives: '{current_task.specific_improvement_directives}'")

    # New fields we might want to log for visibility:
    if current_task.target_solution_description:
        logger.info(
            f"  Target solution description: '{current_task.target_solution_description[:100]}...'")  # Log a snippet
    if current_task.evaluation_guidelines_for_llm_judge:
        logger.info(
            f"  Evaluation guidelines for LLM judge provided (length: {len(current_task.evaluation_guidelines_for_llm_judge)} chars).")

    # 4. Initialize the Task Manager Agent
    try:
        task_manager = TaskManagerAgent(task_definition=current_task)
        task_manager = TaskManagerAgent(task_definition=current_task)
    except ValueError as ve:  # E.g. API key not found in CodeGeneratorAgent's init
        logger.error(f"Configuration error during TaskManagerAgent initialization: {ve}", exc_info=True)
        return
    except Exception as e_tm_init:
        logger.error(f"Unexpected error during TaskManagerAgent initialization: {e_tm_init}", exc_info=True)
        return

    # 5. Run the evolutionary process
    overall_start_time_main = time.monotonic()
    best_programs_found = []  # Initialize

    try:
        best_programs_found = await task_manager.execute()
        if best_programs_found:
            logger.info(
                f"Evolutionary run for task '{current_task.id}' completed. Best program(s) IDs: {[p.id for p in best_programs_found]}")
            # Detailed logging of the best program is now handled by MonitoringAgent/TaskManagerAgent
        else:
            logger.info(
                f"Evolutionary run for task '{current_task.id}' completed. No 'best' programs were identified by the process.")

    except Exception as e_exec:  # Catch errors from the evolutionary cycle itself
        logger.error(
            f"A CRITICAL ERROR occurred during the evolutionary process for task '{current_task.id}': {e_exec}",
            exc_info=True)
    finally:
        overall_end_time_main = time.monotonic()
        logger.info(
            f"Total main.py script execution time: {overall_end_time_main - overall_start_time_main:.2f} seconds.")
        logger.info(f"--- OpenAlpha_Evolve Shutting Down ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OpenAlpha_Evolve with a task defined in a YAML configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    parser.add_argument(
        "--task_config_file",
        type=str,
        required=True,
        help="Path to the YAML task configuration file (e.g., task_definitions/my_task.yaml)."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,  # Default to None, so we only override if value is given
        help=f"Override the number of generations (default from settings: {settings.GENERATIONS})."
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=None,  # Default to None
        help=f"Override the population size (default from settings: {settings.POPULATION_SIZE})."
    )
    # You could add more CLI overrides for other 'settings' if needed (e.g., LLM model names)

    # Optional: environment variable for asyncio debug mode
    # if os.getenv('PYTHONASYNCIODEBUG', '0') == '1':
    #     logger.info("Asyncio debug mode is enabled.")
    # else:
    #     logger.debug("Asyncio debug mode is disabled. Set PYTHONASYNCIODEBUG=1 to enable.")

    cli_args = parser.parse_args()

    # Basic check if task file exists before starting asyncio loop
    if not os.path.isfile(cli_args.task_config_file):
        print(f"Error: Task configuration file not found at '{cli_args.task_config_file}'", file=sys.stderr)
        sys.exit(1)  # Exit if config file is missing

    try:
        asyncio.run(run_alpha_evolve(cli_args))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e_top_level:
        logger.critical(f"A top-level unhandled exception occurred in main: {e_top_level}", exc_info=True)