# Version: 1.0.0 - Lumi's Observant Monitor!
import logging
import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.interfaces import MonitoringAgentInterface, BaseAgent, Program  # Added Program
from config import settings

logger = logging.getLogger(__name__)

METRICS_FILE_CSV = "run_metrics.csv"
METRICS_FILE_JSONL = "run_metrics.jsonl"  # Option for JSONL
DEFAULT_METRIC_FIELDS = [
    "timestamp", "generation_number", "population_size", "num_offspring_generated",
    "avg_correctness", "best_correctness", "avg_pylint_score", "best_pylint_score",
    "avg_runtime_ms", "best_runtime_ms", "avg_cyclomatic_complexity", "best_cyclomatic_complexity",
    "avg_maintainability_index", "best_maintainability_index",
    "generation_time_sec", "task_id",
    "llm_api_calls_generation" # <-- NEW FIELD
]


class MonitoringAgent(MonitoringAgentInterface, BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None, metrics_file_path: Optional[str] = None, log_format: str = "jsonl"): # Default to jsonl

        super().__init__(config)
        self.log_format = log_format.lower()
        if self.log_format == "csv":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_CSV
        elif self.log_format == "jsonl":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_JSONL
        else:
            logger.warning(f"Unsupported log_format '{self.log_format}'. Defaulting to CSV. Supported: 'csv', 'jsonl'.")
            self.log_format = "csv"
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_CSV

        logger.info(
            f"MonitoringAgent initialized. Logging metrics to '{self.metrics_file}' in '{self.log_format}' format.")
        self._ensure_metrics_file_header()

    def _ensure_metrics_file_header(self):  # Method_v1.0.0
        if self.log_format == "csv":
            if not os.path.exists(self.metrics_file) or os.path.getsize(self.metrics_file) == 0:
                try:
                    with open(self.metrics_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                        writer.writeheader()
                    logger.info(f"Metrics CSV file header written to {self.metrics_file}")
                except IOError as e:
                    logger.error(f"Failed to write CSV header to {self.metrics_file}: {e}")
        # JSONL doesn't need a header

    async def log_generation_metrics(self, generation_data: Dict[
        str, Any]):  # Method_v1.0.0 (New method focused on generation)
        """
        Logs detailed metrics for a completed generation to a file.
        `generation_data` should be a flat dictionary matching DEFAULT_METRIC_FIELDS.
        """
        logger.debug(f"Received generation data for logging: {generation_data}")

        # Ensure all default fields are present, fill with None if missing
        log_entry = {field: generation_data.get(field) for field in DEFAULT_METRIC_FIELDS}
        log_entry["timestamp"] = datetime.now().isoformat()  # Ensure timestamp is current

        if self.log_format == "csv":
            try:
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                    writer.writerow(log_entry)
                logger.info(
                    f"Generation {log_entry.get('generation_number', 'N/A')} metrics logged to {self.metrics_file} (CSV).")
            except IOError as e:
                logger.error(f"Failed to write generation metrics to CSV {self.metrics_file}: {e}")
            except Exception as e_csv:
                logger.error(f"Unexpected error writing generation metrics to CSV: {e_csv}", exc_info=True)

        elif self.log_format == "jsonl":
            try:
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                logger.info(
                    f"Generation {log_entry.get('generation_number', 'N/A')} metrics logged to {self.metrics_file} (JSONL).")
            except IOError as e:
                logger.error(f"Failed to write generation metrics to JSONL {self.metrics_file}: {e}")
            except Exception as e_jsonl:
                logger.error(f"Unexpected error writing generation metrics to JSONL: {e_jsonl}", exc_info=True)

        # Also print a summary to console
        await self.report_generation_summary_to_console(log_entry)

    async def report_generation_summary_to_console(self, generation_data: Dict[
        str, Any]):  # Method_v1.0.0 (New method for console summary)
        gen = generation_data.get('generation_number', 'N/A')
        task_id = generation_data.get('task_id', 'N/A')
        avg_correct = generation_data.get('avg_correctness', float('nan')) * 100
        best_correct = generation_data.get('best_correctness', float('nan')) * 100
        avg_pylint = generation_data.get('avg_pylint_score', float('nan'))
        best_pylint = generation_data.get('best_pylint_score', float('nan'))
        gen_time = generation_data.get('generation_time_sec', float('nan'))

        summary_message = (
            f"--- Generation {gen} (Task: {task_id}) Summary ---\n"
            f"  Avg Correctness: {avg_correct:.2f}% | Best Correctness: {best_correct:.2f}%\n"
            f"  Avg Pylint: {avg_pylint:.2f} | Best Pylint: {best_pylint:.2f}\n"
            f"  Generation Time: {gen_time:.2f}s"
        )
        logger.info(summary_message)  # Use logger.info for console output via root logger

    async def log_final_summary(self, best_program_overall: Optional[Program], total_runtime_sec: float,
                                task_id: str):  # Method_v1.0.0
        logger.info("--- Evolutionary Run Final Summary ---")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Total Run Time: {total_runtime_sec:.2f} seconds")
        if best_program_overall:
            logger.info(f"Best Program ID: {best_program_overall.id}")
            logger.info(f"  Generation: {best_program_overall.generation}")
            logger.info(f"  Creation Method: {best_program_overall.creation_method}")
            logger.info(f"  Fitness Scores: {best_program_overall.fitness_scores}")
            logger.info(f"  Code:\n{best_program_overall.code}")
        else:
            logger.info("No successful program was evolved to be deemed 'best overall'.")

    # Deprecating old methods from interface for clarity, new ones are more specific
    async def log_metrics(self, metrics: Dict):  # Deprecated in favor of log_generation_metrics
        logger.warning("MonitoringAgent.log_metrics() is deprecated. Use log_generation_metrics() instead.")
        # For backward compatibility or general use, could adapt it:
        # generation_data = {"generation_number": metrics.get("generation", "unknown"), **metrics}
        # await self.log_generation_metrics(generation_data)
        pass

    async def report_status(self):  # Deprecated in favor of report_generation_summary_to_console
        logger.warning(
            "MonitoringAgent.report_status() is deprecated. Generation summaries are now logged automatically.")
        pass

    async def execute(self, action: str, payload: Optional[Dict] = None) -> Any:  # Method_v1.0.0 (Updated)
        if action == "log_generation_metrics" and payload:
            await self.log_generation_metrics(payload)
            return {"status": "generation metrics logged"}
        elif action == "log_final_summary" and payload:
            await self.log_final_summary(
                best_program_overall=payload.get("best_program_overall"),
                total_runtime_sec=payload.get("total_runtime_sec", 0),
                task_id=payload.get("task_id", "unknown_task")
            )
            return {"status": "final summary logged"}
        else:
            logger.warning(f"Unknown action '{action}' or missing payload for MonitoringAgent.execute().")
            return {"status": f"unknown action '{action}' or missing data"}