import logging
import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.interfaces import MonitoringAgentInterface, BaseAgent, Program
from config import settings

logger = logging.getLogger(__name__)

METRICS_FILE_CSV = "run_metrics.csv"
METRICS_FILE_JSONL = "run_metrics.jsonl"
# --- MODIFIED: DEFAULT_METRIC_FIELDS (v1.1.0 for Ruff) ---
DEFAULT_METRIC_FIELDS = [
    "timestamp", "generation_number", "task_id", "population_size",
    "num_offspring_generated", "generation_time_sec",
    "llm_api_calls_generation",  # Keep this, it's general
    "avg_correctness", "best_correctness",  # Correctness is still key
    # --- Ruff-specific metrics ---
    "avg_ruff_violations",  # Average number of Ruff violations (lower is better)
    "min_ruff_violations",  # Minimum Ruff violations in the generation (best case, lower is better)
    # --- Removing Pylint ---
    # "avg_pylint_score", "best_pylint_score",
    # --- Keeping other metrics ---
    "avg_runtime_ms", "best_runtime_ms",
    "avg_cyclomatic_complexity", "best_cyclomatic_complexity",
    "avg_maintainability_index", "best_maintainability_index",
]


class MonitoringAgent(MonitoringAgentInterface, BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None, metrics_file_path: Optional[str] = None,
                 log_format: str = "jsonl"):
        super().__init__(config)
        self.log_format = log_format.lower()
        if self.log_format == "csv":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_CSV  # METRICS_FILE_CSV needs to be defined, e.g., "run_metrics.csv"
        elif self.log_format == "jsonl":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_JSONL  # METRICS_FILE_JSONL needs to be defined, e.g., "run_metrics.jsonl"
        else:
            logger.warning(
                f"Unsupported log_format '{self.log_format}'. Defaulting to jsonl. Supported: 'csv', 'jsonl'.")
            self.log_format = "jsonl"
            self.metrics_file = metrics_file_path if metrics_file_path else "run_metrics.jsonl"  # Default definition

        logger.info(
            f"MonitoringAgent initialized. Logging metrics to '{self.metrics_file}' in '{self.log_format}' format.")
        self._ensure_metrics_file_header()

    def _ensure_metrics_file_header(self):  # Method_v1.0.1 (No change needed if DEFAULT_METRIC_FIELDS is updated)
        if self.log_format == "csv":
            # Define METRICS_FILE_CSV if not already at module level
            global METRICS_FILE_CSV
            if 'METRICS_FILE_CSV' not in globals():
                METRICS_FILE_CSV = "run_metrics.csv"  # Define it if not present

            if not os.path.exists(self.metrics_file) or os.path.getsize(self.metrics_file) == 0:
                try:
                    with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                        writer.writeheader()
                    logger.info(f"Metrics CSV file header written to {self.metrics_file}")
                except IOError as e:
                    logger.error(f"Failed to write CSV header to {self.metrics_file}: {e}")
        # JSONL doesn't need a header

    async def log_generation_metrics(self, generation_data: Dict[
        str, Any]):  # Method_v1.0.1 (No change needed if data keys match new DEFAULT_METRIC_FIELDS)
        logger.debug(f"Received generation data for logging: {generation_data}")

        log_entry = {field: generation_data.get(field) for field in DEFAULT_METRIC_FIELDS}
        # Ensure timestamp is current and correctly formatted for JSON/CSV
        current_ts = datetime.now()
        log_entry["timestamp"] = current_ts.isoformat() if self.log_format == "jsonl" else current_ts.strftime(
            "%Y-%m-%d %H:%M:%S")

        if self.log_format == "csv":
            try:
                # File should be opened in append mode, and DictWriter will use fieldnames from DEFAULT_METRIC_FIELDS
                with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                    # If the file is empty (e.g. just created by _ensure_metrics_file_header OR if header was missing for some reason)
                    # write the header. This is a bit redundant if _ensure_metrics_file_header always works.
                    # A more robust way is to check if tell() is 0 after opening in 'a' mode.
                    f.seek(0, os.SEEK_END)  # Go to end of file
                    if f.tell() == 0:  # File is empty
                        writer.writeheader()
                    writer.writerow(log_entry)
                logger.info(
                    f"Generation {log_entry.get('generation_number', 'N/A')} metrics logged to {self.metrics_file} (CSV).")
            except IOError as e:
                logger.error(f"Failed to write generation metrics to CSV {self.metrics_file}: {e}")
            except Exception as e_csv:
                logger.error(f"Unexpected error writing generation metrics to CSV: {e_csv}", exc_info=True)

        elif self.log_format == "jsonl":
            try:
                with open(self.metrics_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
                logger.info(
                    f"Generation {log_entry.get('generation_number', 'N/A')} metrics logged to {self.metrics_file} (JSONL).")
            except IOError as e:
                logger.error(f"Failed to write generation metrics to JSONL {self.metrics_file}: {e}")
            except Exception as e_jsonl:
                logger.error(f"Unexpected error writing generation metrics to JSONL: {e_jsonl}", exc_info=True)

        await self.report_generation_summary_to_console(log_entry)

    # --- MODIFIED: report_generation_summary_to_console (v1.1.0 for Ruff) ---
    async def report_generation_summary_to_console(self, generation_data: Dict[str, Any]):  # Method_v1.1.0
        gen = generation_data.get('generation_number', 'N/A')
        task_id = generation_data.get('task_id', 'N/A')
        avg_correct = generation_data.get('avg_correctness', float('nan'))
        best_correct = generation_data.get('best_correctness', float('nan'))

        # Ruff metrics (lower is better)
        avg_ruff_v = generation_data.get('avg_ruff_violations', float('nan'))
        min_ruff_v = generation_data.get('min_ruff_violations', float('nan'))

        gen_time = generation_data.get('generation_time_sec', float('nan'))
        llm_calls = generation_data.get('llm_api_calls_generation', 'N/A')

        summary_message = (
            f"--- Generation {gen} (Task: {task_id}) Summary ---\n"
            f"  LLM Calls this Gen: {llm_calls}\n"
            f"  Avg Correctness: {avg_correct * 100:.2f}% | Best Correctness: {best_correct * 100:.2f}%\n"
            f"  Avg Ruff Violations: {avg_ruff_v:.2f} | Min Ruff Violations: {min_ruff_v:.0f}\n"  # Show Ruff violations
            f"  Generation Time: {gen_time:.2f}s"
        )
        logger.info(summary_message)

    async def log_final_summary(self, best_program_overall: Optional[Program], total_runtime_sec: float,
                                task_id: str,
                                total_llm_api_calls_session: int):  # Method_v1.0.1 (added total_llm_api_calls_session)
        logger.info("--- Evolutionary Run Final Summary ---")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Total Run Time: {total_runtime_sec:.2f} seconds")
        logger.info(f"Total LLM API Calls (Session): {total_llm_api_calls_session}")
        if best_program_overall:
            logger.info(f"Best Program ID: {best_program_overall.id}")
            logger.info(f"  Generation: {best_program_overall.generation}")
            logger.info(f"  Creation Method: {best_program_overall.creation_method}")
            # Display key fitness scores, including Ruff violations
            fitness_summary = {
                "correctness": f"{best_program_overall.fitness_scores.get('correctness', 0.0) * 100:.2f}%",
                "ruff_violations": best_program_overall.fitness_scores.get('ruff_violations', 'N/A'),
                "runtime_ms": f"{best_program_overall.fitness_scores.get('runtime_ms', 'N/A'):.2f}ms"
            }
            logger.info(f"  Fitness Summary: {fitness_summary}")
            logger.info(f"  Code:\n{best_program_overall.code}")  # Be careful logging full code if it can be very long
        else:
            logger.info("No successful program was evolved to be deemed 'best overall'.")

    async def log_metrics(self, metrics: Dict):  # Deprecated
        logger.warning("MonitoringAgent.log_metrics() is deprecated. Use log_generation_metrics() instead.")
        pass

    async def report_status(self):  # Deprecated
        logger.warning(
            "MonitoringAgent.report_status() is deprecated. Generation summaries are now logged automatically.")
        pass

    async def execute(self, action: str,
                      payload: Optional[Dict] = None) -> Any:  # Method_v1.0.1 (Updated for total_llm_api_calls_session)
        if action == "log_generation_metrics" and payload:
            await self.log_generation_metrics(payload)
            return {"status": "generation metrics logged"}
        elif action == "log_final_summary" and payload:
            await self.log_final_summary(
                best_program_overall=payload.get("best_program_overall"),
                total_runtime_sec=payload.get("total_runtime_sec", 0),
                task_id=payload.get("task_id", "unknown_task"),
                total_llm_api_calls_session=payload.get("total_llm_api_calls_session", 0)  # Pass this through
            )
            return {"status": "final summary logged"}
        else:
            logger.warning(f"Unknown action '{action}' or missing payload for MonitoringAgent.execute().")
            return {"status": f"unknown action '{action}' or missing data"}