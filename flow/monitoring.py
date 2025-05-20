# monitoring_agent/monitoring.py
# Version: 1.2.0 (Adding LLM Judge Score Logging)

import logging
import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.interfaces import MetricsLoggerInterface, BaseAgent, Program
from config import settings  # For default metric values if needed

logger = logging.getLogger(__name__)

METRICS_FILE_CSV = "run_metrics.csv"
METRICS_FILE_JSONL = "run_metrics.jsonl"

# --- MODIFIED: DEFAULT_METRIC_FIELDS (Blueprint Step 6) ---
DEFAULT_METRIC_FIELDS = [
    "timestamp", "generation_number", "task_id", "population_size",
    "num_offspring_generated", "generation_time_sec",
    "llm_api_calls_generation",
    "avg_correctness", "best_correctness",
    "avg_ruff_violations", "min_ruff_violations",
    # --- NEW LLM Judge Score Metrics ---
    "avg_llm_judge_score",  # Average LLM judge score for the generation
    "best_llm_judge_score",  # Best LLM judge score in the generation
    # --- End NEW ---
    "avg_runtime_ms", "best_runtime_ms",
    "avg_cyclomatic_complexity", "best_cyclomatic_complexity",
    "avg_maintainability_index", "best_maintainability_index",
]


class MetricsLogger(MetricsLoggerInterface, BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None, metrics_file_path: Optional[str] = None,
                 log_format: str = "jsonl"):
        super().__init__(config)
        self.log_format = log_format.lower()
        if self.log_format == "csv":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_CSV
        elif self.log_format == "jsonl":
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_JSONL
        else:
            logger.warning(
                f"Unsupported log_format '{self.log_format}'. Defaulting to jsonl. Supported: 'csv', 'jsonl'.")
            self.log_format = "jsonl"
            self.metrics_file = metrics_file_path if metrics_file_path else METRICS_FILE_JSONL

        logger.info(
            f"MetricsLogger initialized. Logging metrics to '{self.metrics_file}' in '{self.log_format}' format.")
        self._ensure_metrics_file_header()

    def _ensure_metrics_file_header(self):  # Method_v1.0.1 (No change needed as it uses updated DEFAULT_METRIC_FIELDS)
        if self.log_format == "csv":
            if not os.path.exists(self.metrics_file) or os.path.getsize(self.metrics_file) == 0:
                try:
                    with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                        writer.writeheader()
                    logger.info(
                        f"Metrics CSV file header written to {self.metrics_file} (including LLM judge score fields).")
                except IOError as e:
                    logger.error(f"Failed to write CSV header to {self.metrics_file}: {e}")
        # JSONL doesn't need a header

    async def log_generation_metrics(self, generation_data: Dict[
        str, Any]):  # Method_v1.1.0 (Handles new fields via DEFAULT_METRIC_FIELDS)
        logger.debug(f"Received generation data for logging: {generation_data}")

        # Ensure all default fields are present, using None or a suitable default if missing in generation_data
        log_entry = {}
        for field in DEFAULT_METRIC_FIELDS:
            if field == "timestamp":  # Timestamp is handled separately
                continue
            # Use a specific default for metrics if they might be missing and need one
            default_val_for_metric = 0.0  # Or float('nan'), or None, depending on how dashboard handles it
            if field in ["avg_ruff_violations", "min_ruff_violations", "avg_runtime_ms", "best_runtime_ms",
                         "avg_cyclomatic_complexity", "best_cyclomatic_complexity"]:
                default_val_for_metric = float('inf')
            elif field in ["avg_llm_judge_score", "best_llm_judge_score"]:
                # Default LLM judge score could be 0 or a specific "not available" marker if needed
                default_val_for_metric = settings.DEFAULT_METRIC_VALUE.get("llm_judge_overall_score", 0.0)

            log_entry[field] = generation_data.get(field, default_val_for_metric)

        current_ts = datetime.now()
        log_entry["timestamp"] = current_ts.isoformat() if self.log_format == "jsonl" else current_ts.strftime(
            "%Y-%m-%d %H:%M:%S")

        if self.log_format == "csv":
            try:
                with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_METRIC_FIELDS)
                    f.seek(0, os.SEEK_END)
                    if f.tell() == 0:
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

        await self.print_gen_summary(log_entry)  # Pass the processed log_entry

    # --- MODIFIED: print_gen_summary (Blueprint Step 6) ---
    async def print_gen_summary(self, generation_data: Dict[str, Any]):  # Method_v1.2.0
        gen = generation_data.get('generation_number', 'N/A')
        task_id = generation_data.get('task_id', 'N/A')
        avg_correct = generation_data.get('avg_correctness', float('nan'))
        best_correct = generation_data.get('best_correctness', float('nan'))
        avg_ruff_v = generation_data.get('avg_ruff_violations', float('nan'))
        min_ruff_v = generation_data.get('min_ruff_violations', float('nan'))

        # --- NEW: LLM Judge Scores ---
        avg_llm_score = generation_data.get('avg_llm_judge_score', float('nan'))
        best_llm_score = generation_data.get('best_llm_judge_score', float('nan'))
        # --- END NEW ---

        gen_time = generation_data.get('generation_time_sec', float('nan'))
        llm_calls = generation_data.get('llm_api_calls_generation', 'N/A')

        summary_lines = [
            f"--- Generation {gen} (Task: {task_id}) Summary ---",
            f"  LLM Calls this Gen: {llm_calls}",
            f"  Avg Correctness: {avg_correct * 100:.2f}% | Best Correctness: {best_correct * 100:.2f}%"
        ]
        # Only add LLM judge scores if they are not NaN (i.e., data was available)
        if not (isinstance(avg_llm_score, float) and avg_llm_score != avg_llm_score):  # Check for NaN
            summary_lines.append(
                f"  Avg LLM Judge Score: {avg_llm_score:.2f}/10 | Best LLM Judge Score: {best_llm_score:.2f}/10")

        summary_lines.append(f"  Avg Ruff Violations: {avg_ruff_v:.2f} | Min Ruff Violations: {min_ruff_v:.0f}")
        summary_lines.append(f"  Generation Time: {gen_time:.2f}s")

        logger.info("\n".join(summary_lines))

    async def log_run_summary(self, best_program_overall: Optional[Program], total_runtime_sec: float,
                                task_id: str,
                                total_llm_api_calls_session: int):  # Method_v1.1.0 (Added llm_judge_overall_score to summary)
        logger.info("--- Evolutionary Run Final Summary ---")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Total Run Time: {total_runtime_sec:.2f} seconds")
        logger.info(f"Total LLM API Calls (Session): {total_llm_api_calls_session}")
        if best_program_overall:
            logger.info(f"Best Program ID: {best_program_overall.id}")
            logger.info(f"  Generation: {best_program_overall.generation}")
            logger.info(f"  Creation Method: {best_program_overall.creation_method}")

            fitness_summary = {
                "correctness": f"{best_program_overall.fitness_scores.get('correctness', 0.0) * 100:.2f}%",
                "ruff_violations": best_program_overall.fitness_scores.get('ruff_violations', 'N/A'),
                "runtime_ms": f"{best_program_overall.fitness_scores.get('runtime_ms', 'N/A'):.2f}ms",
                # --- NEW: Add LLM Judge score to final summary ---
                "llm_judge_overall_score": f"{best_program_overall.fitness_scores.get('llm_judge_overall_score', 'N/A')}/10"
            }
            logger.info(f"  Fitness Summary: {fitness_summary}")
            if best_program_overall.ai_feedback:  # Also log the feedback if available
                logger.info(f"  LLM Judge Feedback: {best_program_overall.ai_feedback}")
            # logger.info(f"  Code:\n{best_program_overall.code}") # Careful with long code
        else:
            logger.info("No successful program was evolved to be deemed 'best overall'.")

    async def log_metrics(self, metrics: Dict):
        logger.warning("MetricsLogger.log_metrics() is deprecated. Use log_generation_metrics() instead.")
        pass

    async def report_status(self):
        logger.warning(
            "MetricsLogger.report_status() is deprecated. Generation summaries are now logged automatically.")
        pass

    async def execute(self, action: str,
                      payload: Optional[Dict] = None) -> Any:
        if action == "log_generation_metrics" and payload:
            await self.log_generation_metrics(payload)
            return {"status": "generation metrics logged"}
        elif action == "log_run_summary" and payload:
            await self.log_run_summary(
                best_program_overall=payload.get("best_program_overall"),
                total_runtime_sec=payload.get("total_runtime_sec", 0),
                task_id=payload.get("task_id", "unknown_task"),
                total_llm_api_calls_session=payload.get("total_llm_api_calls_session", 0)
            )
            return {"status": "final summary logged"}
        else:
            logger.warning(f"Unknown action '{action}' or missing payload for MetricsLogger.execute().")
            return {"status": f"unknown action '{action}' or missing data"}