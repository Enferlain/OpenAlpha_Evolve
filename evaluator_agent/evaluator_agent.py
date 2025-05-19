# evaluator_agent/evaluator_agent.py
# Version: 1.2.0 (Adding standalone script execution check)

import time
import logging
import traceback
import subprocess
import tempfile
import os
import ast
import copy
import re
import json
import asyncio
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any, Tuple, Union, List
# from pylint import lint # REMOVED
# from pylint.lint import PyLinter # REMOVED
# from pylint.reporters.text import TextReporter # REMOVED
from core.interfaces import (
    EvaluatorAgentInterface, Program, TaskDefinition,
    PromptDesignerInterface, CodeGeneratorInterface # NEW: For calling judge LLM
)
from config import settings

logger = logging.getLogger(__name__)


# Method_v1.4.0 ("Integrate Ruff for Linting")
def run_ruff_in_thread(file_path: str, project_root: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Runs Ruff on the given file and returns the list of violations and their count.
    Ruff is expected to be configured via pyproject.toml in the project_root.

    Args:
        file_path: The absolute path to the temporary Python script to lint.
        project_root: The absolute path to the project root where pyproject.toml is located.

    Returns:
        A tuple containing:
        - A list of violation objects (dictionaries) from Ruff's JSON output.
        - An integer count of the total violations found.
    """
    logger.info(f"Running Ruff analysis for program file: {file_path}")
    logger.debug(f"Project root for Ruff (pyproject.toml lookup): {project_root}")

    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--output-format=json",
        "--no-cache",
        file_path
    ]

    violations_list: List[Dict[str, Any]] = []
    violations_count = 0

    try:
        logger.debug(f"Executing Ruff command: {' '.join(command)}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            cwd=project_root
        )

        stdout_str = process.stdout.strip()
        stderr_str = process.stderr.strip()

        if stderr_str:
            logger.warning(f"Ruff stderr output for {file_path}:\n{stderr_str}")

        if stdout_str:
            logger.debug(f"Ruff stdout (JSON output) for {file_path}:\n{stdout_str}")
            try:
                parsed_json = json.loads(stdout_str)
                if isinstance(parsed_json, list):
                    violations_list = parsed_json
                    violations_count = len(violations_list)
                    logger.info(f"Ruff found {violations_count} issues for {file_path}.")
                else:
                    logger.error(f"Ruff JSON output was not a list as expected for {file_path}. Output: {stdout_str}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode Ruff JSON output for {file_path}: {e}. Raw output: '{stdout_str}'")
        else:
            logger.warning(f"Ruff produced no stdout for {file_path}. Exit code: {process.returncode}. Stderr: {stderr_str}")

        if process.returncode > 1:
            logger.error(f"Ruff execution failed with exit code {process.returncode} for {file_path}. Stderr: {stderr_str}")

    except subprocess.TimeoutExpired:
        logger.error(f"Ruff execution timed out for {file_path}.")
    except FileNotFoundError:
        logger.error(f"Ruff command not found. Is `ruff` installed and in PATH or `sys.executable -m ruff` working? Command: {' '.join(command)}")
    except Exception as e_ruff:
        logger.error(f"An unexpected error occurred during Ruff execution for {file_path}: {e_ruff}", exc_info=True)

    logger.debug(f"Ruff analysis for {file_path} completed. Violations found: {violations_count}")
    return violations_list, violations_count

# --- NEW: Helper to format execution/Ruff feedback for LLM Judge prompt ---
def _format_summary_for_llm_judge(self, program: Program, max_errors_to_show: int = 3) -> Tuple[str, str]:
    """
    Formats execution errors/logs and Ruff issues for the LLM Judge prompt.
    Returns (formatted_execution_summary, formatted_ruff_summary)
    """
    # Execution Summary
    exec_feedback_lines = []
    execution_related_errors = [
        e for e in program.errors
        if "Execution Log (STDOUT" in e or \
           "Execution Log (STDERR" in e or \
           "Standalone script execution failed" in e or \
           "ExecutionFailure: ImportError" in e or # Specific import error
           "I/O Test Harness Error" in e # Error from I/O harness
    ]

    if execution_related_errors:
        for i, err_log in enumerate(execution_related_errors):
            if i < max_errors_to_show:
                # Remove potential multiple newlines for conciseness in prompt
                exec_feedback_lines.append(f"  - {err_log.replace(chr(10)+chr(10), chr(10)).strip()}")
            else:
                exec_feedback_lines.append(f"  ... and {len(execution_related_errors) - max_errors_to_show} more execution-related messages.")
                break
    else:
        exec_feedback_lines.append("  - No specific execution issues or logs captured beyond run status.")
    formatted_execution_summary = "\n".join(exec_feedback_lines)

    # Ruff Summary
    ruff_feedback_lines = []
    ruff_issues = [e for e in program.errors if e.startswith("Ruff-")]
    if ruff_issues:
        for i, ruff_issue in enumerate(ruff_issues):
            if i < max_errors_to_show:
                ruff_feedback_lines.append(f"  - {ruff_issue.strip()}")
            else:
                ruff_feedback_lines.append(f"  ... and {len(ruff_issues) - max_errors_to_show} more Ruff issues.")
                break
    else:
        ruff_feedback_lines.append("  - No Ruff issues reported.")
    formatted_ruff_summary = "\n".join(ruff_feedback_lines)

    return formatted_execution_summary, formatted_ruff_summary

# --- NEW HELPER METHOD for Standalone Script Execution (Blueprint Step 2) ---
async def _execute_standalone_script(
    self,
    program_id: str, # For logging
    code: str,
    timeout_seconds: Optional[int] = None
) -> Tuple[bool, str, str, Optional[float]]: # (runs_ok, stdout, stderr, execution_time_ms)
    """
    Executes the given code as a standalone Python script in a temporary file.
    Returns whether it ran successfully (exit code 0), stdout, stderr, and execution time.
    """
    effective_timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
    logger.debug(f"[_execute_standalone_script] Attempting to run program ID: {program_id} as standalone script. Timeout: {effective_timeout}s.")

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"script_{program_id}.py")

    try:
        with open(temp_file_path, "w", encoding='utf-8') as f:
            f.write(code)
    except Exception as e_write:
        logger.error(f"[_execute_standalone_script] Failed to write code for {program_id} to temp file {temp_file_path}: {e_write}")
        return False, "", f"Error writing script to temp file: {e_write}", None

    cmd = [sys.executable, temp_file_path]
    proc = None
    start_time_exec = 0.0
    end_time_exec = 0.0

    try:
        logger.debug(f"[_execute_standalone_script] Executing: {' '.join(cmd)} in {temp_dir}")
        start_time_exec = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=temp_dir # Run script from its own temp directory
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        end_time_exec = time.monotonic()
        execution_time_ms = (end_time_exec - start_time_exec) * 1000

        stdout_str = stdout_bytes.decode('utf-8', errors='replace').strip()
        stderr_str = stderr_bytes.decode('utf-8', errors='replace').strip()

        logger.debug(f"[_execute_standalone_script] {program_id} finished. RC: {proc.returncode}, Time: {execution_time_ms:.2f}ms")
        if stdout_str: logger.debug(f"[_execute_standalone_script] {program_id} STDOUT:\n{stdout_str}")
        if stderr_str: logger.debug(f"[_execute_standalone_script] {program_id} STDERR:\n{stderr_str}")

        runs_ok = proc.returncode == 0
        return runs_ok, stdout_str, stderr_str, execution_time_ms

    except asyncio.TimeoutError:
        if proc:
            try: proc.kill(); await proc.wait()
            except ProcessLookupError: pass
            except Exception as e_kill: logger.error(f"Error killing timed-out process for {program_id}: {e_kill}")
        logger.warning(f"[_execute_standalone_script] {program_id} execution timed out after {effective_timeout}s.")
        return False, "", f"Execution timed out after {effective_timeout} seconds.", (time.monotonic() - start_time_exec) * 1000
    except Exception as e_exec:
        logger.error(f"[_execute_standalone_script] Unexpected error executing {program_id}: {e_exec}", exc_info=True)
        exec_time = (time.monotonic() - start_time_exec) * 1000 if start_time_exec > 0 else None
        return False, "", f"Unexpected execution error: {type(e_exec).__name__} - {str(e_exec)}", exec_time
    finally:
        try:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
            if os.path.exists(temp_dir): os.rmdir(temp_dir)
        except Exception as e_cleanup:
            logger.error(f"Error cleaning temp files for {program_id}: {e_cleanup}")
# --- END NEW HELPER METHOD ---


class EvaluatorAgent(EvaluatorAgentInterface):
    def __init__(self,
                 task_definition: Optional[TaskDefinition] = None,
                 prompt_designer: Optional[PromptDesignerInterface] = None, # NEW
                 code_generator_for_judge: Optional[CodeGeneratorInterface] = None # NEW
                ):
        super().__init__(config=None)
        self.task_definition = task_definition # For overall context
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS

        # Agents needed for LLM-as-Judge functionality
        # These could be passed in or instantiated here. For now, let's assume they can be set up.
        # In TaskManagerAgent, these would be passed from its own instances.
        self.prompt_designer = prompt_designer
        self.code_generator_for_judge = code_generator_for_judge

        # Judge LLM model name (can be different from code generation model)
        self.judge_llm_model_name = settings.EVALUATION_MODEL_NAME # From settings.py
        self.judge_llm_temperature = 0.3 # Typically lower temp for more factual/consistent judging

        logger.info(f"EvaluatorAgent initialized. Timeout: {self.evaluation_timeout_seconds}s. Judge Model: {self.judge_llm_model_name}")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition ID: {self.task_definition.id}")
        if not self.prompt_designer or not self.code_generator_for_judge:
            logger.warning("EvaluatorAgent initialized WITHOUT prompt_designer or code_generator_for_judge. LLM-as-Judge functionality will be disabled.")

    # Add the new helper method to the class
    _execute_standalone_script = _execute_standalone_script # Binding the async def

    def _check_syntax(self, code: str) -> List[str]:
        logger.debug(f"[_check_syntax] Checking syntax for code (first 100 chars): {code[:100].replace(chr(10), '<NL>')}") # <NL> for newlines
        errors = []
        try:
            ast.parse(code)
            logger.debug(f"[_check_syntax] Syntax OK.")
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}")
            logger.debug(f"[_check_syntax] SyntaxError found: {errors[-1]}")
        except Exception as e:
            errors.append(f"Unexpected error during syntax check: {str(e)}")
            logger.debug(f"[_check_syntax] Unexpected syntax check error: {errors[-1]}")
        return errors

    async def _execute_code_safely(
        self,
        code: str,
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        logger.debug(f"[_execute_code_safely] Starting for func: {task_for_examples.function_name_to_evolve}, timeout: {timeout}s. Code (first 100): {code[:100].replace(chr(10), '<NL>')}")

        results = {"test_outputs": [], "average_runtime_ms": 0.0}

        if not task_for_examples.input_output_examples:
            logger.warning("No input/output examples provided to _execute_code_safely.")
            return results, "No test cases to run."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task {task_for_examples.id} does not specify 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        serializable_test_cases = copy.deepcopy(task_for_examples.input_output_examples)

        def prep_for_json(obj):
            if isinstance(obj, dict):
                return {str(k): prep_for_json(v) for k, v in obj.items()}  # Stringify keys for top-level graph obj
            elif isinstance(obj, list):
                return [prep_for_json(elem) for elem in obj]
            elif obj == float('inf'):
                return "Infinity"  # Placeholder string
            elif obj == float('-inf'):
                return "-Infinity"  # Placeholder string
            elif isinstance(obj, float) and obj != obj:  # NaN
                return "NaN"  # Placeholder string
            return obj

        # This part is tricky. We want to serialize the graph in a way that the harness can reconstruct it
        # with integer keys if they were originally integers.
        # The issue is json.dumps *always* makes keys strings.
        # So, the HARNESS must do the reconversion.

        # Let's just dump it as is, and let the harness fix the keys if they are graph keys.
        test_cases_str_for_harness = json.dumps(serializable_test_cases)
        # The replacements for Infinity/NaN need to be done carefully so they become float('inf') etc. in Python
        test_cases_str_for_harness = test_cases_str_for_harness.replace('"Infinity"', 'float("inf")')
        test_cases_str_for_harness = test_cases_str_for_harness.replace('"-Infinity"', 'float("-inf")')
        test_cases_str_for_harness = test_cases_str_for_harness.replace('"NaN"', 'float("nan")')

        test_harness_code = f"""
import json
import time
import sys
import math

# User's code (function to be tested)
{code}

# --- Helper function to convert string keys back to int if possible ---
def int_keys_if_possible(obj):
    if isinstance(obj, dict):
        new_dict = {{}}
        for k, v in obj.items():
            try:
                # Try to convert key to int. If it's not an int-like string, keep original.
                # This is a simple heuristic for graph node IDs.
                # More robust: check if all keys in a specific 'graph' dict are int-like.
                int_k = int(k)
                new_dict[int_k] = int_keys_if_possible(v)
            except ValueError:
                new_dict[k] = int_keys_if_possible(v) # Keep original key if not int-like
        return new_dict
    elif isinstance(obj, list):
        return [int_keys_if_possible(elem) for elem in obj]
    return obj
# --- End Helper ---

results = []
total_execution_time = 0
num_tests = 0

Infinity = float('inf')
NaN = float('nan')

# The test_cases string from json.dumps will have string keys for graphs.
raw_test_cases = {test_cases_str_for_harness} 
function_to_test_name = "{task_for_examples.function_name_to_evolve}"

# Make sure the function_to_test is available in the global scope
if function_to_test_name not in globals():
    # Attempt to find it if it was defined inside a class (common for LLM output)
    # This is a simple heuristic and might need refinement.
    found_func = None
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            if hasattr(obj, function_to_test_name):
                method = getattr(obj, function_to_test_name)
                if callable(method):
                    globals()[function_to_test_name] = method
                    found_func = True
                    break
    if not found_func:
        print(json.dumps({{"error": f"Function '{{function_to_test_name}}' not found in the global scope or as a callable method of a defined class."}}))
        sys.exit(1)
        
function_to_test = globals()[function_to_test_name]

for i, raw_test_case in enumerate(raw_test_cases):
    # Apply key conversion specifically to the 'graph' part of the input if it exists
    input_args = raw_test_case.get("input")
    if isinstance(input_args, dict) and 'graph' in input_args and isinstance(input_args['graph'], dict):
        # This assumes graph node IDs are the primary things needing int keys.
        # If other dicts in input_args need int keys, this logic needs adjustment.
        input_args['graph'] = int_keys_if_possible(input_args['graph'])
        # Also, if the source_node itself was stringified by an outer layer (unlikely here), it would need int()
        if 'source_node' in input_args and isinstance(input_args['source_node'], str):
            try:
                input_args['source_node'] = int(input_args['source_node'])
            except ValueError:
                pass # Keep as string if not int-like
    
    start_time = time.perf_counter()
    try:
        if isinstance(input_args, list):
            actual_output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            actual_output = function_to_test(**input_args)
        elif input_args is None:
             actual_output = function_to_test()
        else:
            actual_output = function_to_test(input_args)
            
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        total_execution_time += execution_time_ms
        num_tests += 1
        results.append({{"test_case_id": i, "output": actual_output, "runtime_ms": execution_time_ms, "status": "success"}})
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        error_output = {{
            "test_case_id": i,
            "error": str(e), 
            "error_type": type(e).__name__,
            "runtime_ms": execution_time_ms,
            "status": "error"
        }}
        try:
            json.dumps(error_output)
        except TypeError:
            error_output["error"] = "Unserializable error object"
        results.append(error_output)

final_output = {{"test_outputs": results}}
if num_tests > 0:
    final_output["average_runtime_ms"] = total_execution_time / num_tests

def custom_json_serializer(obj):
    if isinstance(obj, float):
        if obj == float('inf'):
            return 'Infinity'
        elif obj == float('-inf'):
            return '-Infinity'
        elif obj != obj:
            return 'NaN'
    raise TypeError(f"Object of type {{type(obj).__name__}} is not JSON serializable")

print(json.dumps(final_output, default=custom_json_serializer))
        """
        logger.debug(f"[_execute_code_safely] Generated Test Harness Code:\n{test_harness_code}") # LOG THE FULL HARNESS

        with open(temp_file_path, "w") as f:
            f.write(test_harness_code)

        cmd = [sys.executable, temp_file_path]
        logger.debug(f"[_execute_code_safely] Generated Test Harness Code (with int_keys_if_possible):\n{test_harness_code}")

        proc = None
        try:
            logger.debug(f"Executing code: {' '.join(cmd)} in {temp_dir}")
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.monotonic() - start_time
            logger.debug(f"Code execution finished in {duration:.2f}s. Exit code: {proc.returncode}")

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()
            logger.debug(f"[_execute_code_safely] Subprocess finished. Return Code: {proc.returncode}")
            logger.debug(f"[_execute_code_safely] STDOUT:\n{stdout_str}") # LOG FULL STDOUT
            logger.debug(f"[_execute_code_safely] STDERR:\n{stderr_str}") # LOG FULL STDERR

            if proc.returncode != 0:
                error_message = f"Execution failed with exit code {proc.returncode}. Stdout: '{stdout_str}'. Stderr: '{stderr_str}'"
                logger.warning(f"[_execute_code_safely] {error_message}")
                logger.debug(f"[_execute_code_safely] Returning: (None, '{error_message}') due to non-zero exit code.")
                return None, error_message

            if not stdout_str:
                 error_message = f"No output from script. Stderr: '{stderr_str}'"
                 logger.warning(f"[_execute_code_safely] {error_message}")
                 logger.debug(f"[_execute_code_safely] Returning: (None, '{error_message}') due to no stdout.")
                 return None, error_message

            try:
                def json_loads_with_infinity(s):
                    s = s.replace('"Infinity"', 'float("inf")')
                    s = s.replace('"-Infinity"', 'float("-inf")')
                    s = s.replace('"NaN"', 'float("nan")')
                    return json.loads(s)

                parsed_output = json_loads_with_infinity(stdout_str)
                logger.debug(f"[_execute_code_safely] Parsed execution output: {parsed_output}")
                logger.debug(f"[_execute_code_safely] Returning: (parsed_output, None) successfully.")
                return parsed_output, None
            except json.JSONDecodeError as e:
                error_message = f"Failed to decode JSON output: {e}. Raw output: '{stdout_str}'"
                logger.error(f"[_execute_code_safely] {error_message}")
                logger.debug(f"[_execute_code_safely] Returning: (None, '{error_message}') due to JSONDecodeError.")
                return None, error_message
            except Exception as e:
                error_message = f"Error processing script output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message

        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                except Exception as e_kill:
                    logger.error(f"Error trying to kill timed-out process: {e_kill}")
            error_message = f"Execution timed out after {timeout} seconds."
            logger.warning(f"[_execute_code_safely] {error_message}")
            logger.debug(f"[_execute_code_safely] Returning: (None, '{error_message}') due to TimeoutError.")
            return None, error_message
        except Exception as e:
            error_message = f"Unexpected execution error: {str(e)}"
            logger.error(f"[_execute_code_safely] An unexpected error occurred: {e}", exc_info=True)
            logger.debug(f"[_execute_code_safely] Returning: (None, '{error_message}') due to unexpected Exception.")
            return None, error_message
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e_cleanup:
                logger.error(f"Error during cleanup of temp files: {e_cleanup}")
            logger.debug(f"[_execute_code_safely] Cleanup of temp files attempted for {temp_file_path}")

    def _assess_correctness(self, execution_results: Dict[str, Any], expected_outputs: List[Dict[str, Any]]) -> Tuple[
        float, int, int]:
        passed_tests = 0
        total_tests = len(expected_outputs)

        if not execution_results or "test_outputs" not in execution_results:
            logger.warning("[_assess_correctness] Execution results are missing 'test_outputs' field.")
            return 0.0, 0, total_tests

        actual_test_outputs_from_json = execution_results["test_outputs"]

        if len(actual_test_outputs_from_json) != total_tests:
            logger.warning(
                f"[_assess_correctness] Mismatch in number of test outputs ({len(actual_test_outputs_from_json)}) and expected outputs ({total_tests}).")

        for i, expected_case in enumerate(expected_outputs):
            actual_output_detail_json = next(
                (res for res in actual_test_outputs_from_json if res.get("test_case_id") == i), None)

            if actual_output_detail_json and actual_output_detail_json.get("status") == "success":
                actual_output_from_llm_json = actual_output_detail_json.get("output")
                expected_val_from_yaml = expected_case["output"]  # This has int keys from PyYAML

                # --- NEW: Convert keys of actual_output_from_llm_json to int if they are dicts ---
                # This assumes the output structure is a flat dictionary (like Dijkstra's output)
                # or that we only care about the top-level keys if it's nested.
                # For Dijkstra, the output is Dict[NodeID, Distance].

                processed_actual_output = {}
                if isinstance(actual_output_from_llm_json, dict):
                    for k_str, v_val in actual_output_from_llm_json.items():
                        try:
                            processed_actual_output[int(k_str)] = v_val
                        except ValueError:
                            # If a key cannot be int, it might be a different kind of output, or an error
                            logger.warning(
                                f"[_assess_correctness] Test case {i}: Could not convert actual output key '{k_str}' to int. Keeping as string.")
                            processed_actual_output[k_str] = v_val
                else:  # If actual output isn't a dict, use it as is (e.g. a single value, list)
                    processed_actual_output = actual_output_from_llm_json
                # --- END NEW ---

                # Now compare with integer keys against integer keys
                if processed_actual_output == expected_val_from_yaml:
                    passed_tests += 1
                else:
                    # For debugging, it's useful to see both, and their types
                    logger.debug(f"[_assess_correctness] Test case {i} FAILED:")
                    logger.debug(
                        f"  Expected (type: {type(expected_val_from_yaml)}, keys: {list(expected_val_from_yaml.keys()) if isinstance(expected_val_from_yaml, dict) else 'N/A'}): {expected_val_from_yaml}")
                    logger.debug(
                        f"  Got (type: {type(processed_actual_output)}, keys: {list(processed_actual_output.keys()) if isinstance(processed_actual_output, dict) else 'N/A'}): {processed_actual_output}")
                    # Log the original JSON string version too for good measure if it was a dict
                    if isinstance(actual_output_from_llm_json, dict):
                        logger.debug(f"  Original 'Got' from JSON (str keys): {actual_output_from_llm_json}")

            elif actual_output_detail_json:  # status was not "success"
                logger.debug(
                    f"[_assess_correctness] Test case {i} had error in harness: {actual_output_detail_json.get('error_type')} - {actual_output_detail_json.get('error')}")
            else:  # No output detail found for this test case index
                logger.debug(f"[_assess_correctness] Test case {i}: No output detail found in results.")

        if total_tests == 0:
            logger.debug("[_assess_correctness] No test cases to run, correctness is 1.0 by default.")
            return 1.0, 0, 0  # Or perhaps 0.0 if no tests means it can't be verified? Your choice.

        correctness = passed_tests / total_tests
        logger.debug(
            f"[_assess_correctness] Final assessment: Correctness={correctness:.2f} ({passed_tests}/{total_tests})")
        return correctness, passed_tests, total_tests

    # Method_v1.1.2 (ANSI escape code stripping for Radon output)
    async def _run_static_analysis_tool(self, tool_name: str, command: List[str], program_id: str) -> \
            Tuple[Optional[Dict[str, Any]], Optional[str]]:

        logger.debug(f"Executing {tool_name} command for program {program_id}: {' '.join(command)}")
        try:
            process = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, universal_newlines=True, timeout=60
            )

            raw_stdout = process.stdout  # Get the raw stdout
            stderr = process.stderr.strip()

            # Regex to remove ANSI escape codes
            ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            stdout_cleaned_ansi = ansi_escape_pattern.sub('', raw_stdout)

            # Now strip whitespace from the ANSI-cleaned string
            stdout = stdout_cleaned_ansi.strip()

            if stderr:
                logger.warning(
                    f"{tool_name} produced stderr for {program_id} "
                    f"(return code {process.returncode}):\n{stderr}"
                )

            if process.returncode != 0:
                error_message = (
                    f"{tool_name} for program {program_id} failed with exit code {process.returncode}."
                )
                # Log the *original raw_stdout* if there's an error, before cleaning, to see everything
                if raw_stdout.strip(): error_message += f" Raw Stdout excerpt: '{raw_stdout.strip()[:200]}...'"
                if stderr: error_message += f" Stderr excerpt: '{stderr[:200]}...'"
                logger.error(error_message)
                return None, error_message

            if not stdout:  # Check the cleaned and stripped stdout
                logger.info(
                    f"{tool_name} for program {program_id} succeeded but produced no significant output "
                    f"(cleaned stdout was empty). Raw stdout was: '{raw_stdout[:200]}...'"
                )
                return {}, None  # Return empty dict for no data, but no error

            logger.debug(
                f"Attempting json.loads on ANSI-cleaned and stripped stdout of length: {len(stdout)} for {program_id}")
            if raw_stdout != stdout:  # Log if cleaning made a difference
                logger.debug(f"Original raw stdout (first 200): {repr(raw_stdout[:200])}")
                logger.debug(f"ANSI-cleaned stdout (first 200): {repr(stdout[:200])}")
            try:
                parsed_json = json.loads(stdout)  # Parse the cleaned string
                return parsed_json, None
            except json.JSONDecodeError as e:
                error_char_offset = e.pos
                context_chars = 40  # Increased context

                start_slice = max(0, error_char_offset - context_chars)
                end_slice = min(len(stdout), error_char_offset + context_chars)
                problematic_slice = stdout[start_slice:end_slice]

                logger.error(
                    f"Failed to decode {tool_name} JSON output for {program_id} AFTER ANSI cleaning: {e}. Error at char {error_char_offset}."
                )
                logger.error(f"  Cleaned stdout length: {len(stdout)}")
                logger.error(
                    f"  Problematic slice (from cleaned stdout, around char {error_char_offset}): >>>{repr(problematic_slice)}<<<")
                logger.error(f"  Full cleaned stdout (first 500 chars): >>>{repr(stdout[:500])}<<<")
                if len(stdout) > 500:
                    logger.error(f"  Full cleaned stdout (last 500 chars): >>>{repr(stdout[-500:])}<<<")
                # Also log the original raw stdout for comparison if error persists
                logger.error(
                    f"  Original raw stdout before any cleaning (first 500 chars): >>>{repr(raw_stdout[:500])}<<<")
                return None, f"{tool_name}: Failed to decode JSON output even after ANSI cleaning."

        except subprocess.TimeoutExpired:
            logger.error(f"{tool_name} execution timed out for program {program_id}.")
            return None, f"{tool_name}: Execution timed out."
        except FileNotFoundError:
            logger.error(
                f"{tool_name} command not found (is it installed and in PATH?). "
                f"Failed command: {' '.join(command)}"
            )
            return None, f"{tool_name}: Command not found. Is {tool_name} installed?"
        except Exception as e:
            logger.error(
                f"Unexpected error during {tool_name} analysis for program {program_id}: {e}",
                exc_info=True
            )
            return None, f"{tool_name}: Unspecified analysis error - {type(e).__name__}"

    async def _invoke_llm_judge(self,
                                task: TaskDefinition,
                                program_to_judge: Program,
                                execution_summary_for_judge: Dict[str, Any]
                                ) -> Tuple[Optional[float], Optional[str]]:  # (score, justification)
        """
        Invokes the LLM-as-a-Judge.
        """
        if not self.prompt_designer or not self.code_generator_for_judge:
            logger.warning(
                f"LLM Judge cannot be invoked for {program_to_judge.id}: Missing prompt_designer or code_generator_for_judge.")
            return None, "LLM Judge components not configured."

        if not task.evaluation_guidelines_for_llm_judge:
            logger.info(
                f"No evaluation_guidelines_for_llm_judge for task {task.id}. Skipping LLM Judge for {program_to_judge.id}.")
            return None, "No user guidelines provided for LLM Judge."

        judge_prompt = self.prompt_designer.design_llm_judge_prompt(
            task=task,
            program_to_judge=program_to_judge,
            execution_summary=execution_summary_for_judge
        )

        try:
            logger.info(f"Calling Judge LLM ({self.judge_llm_model_name}) for program {program_to_judge.id}...")
            # Using the code_generator_for_judge instance to make the API call
            raw_judge_response = await self.code_generator_for_judge.generate_code(
                # Using generate_code as it's the generic LLM call method
                prompt=judge_prompt,
                model_name=self.judge_llm_model_name,
                temperature=self.judge_llm_temperature,
                output_format="code"  # Expecting raw text (JSON string)
            )

            if not raw_judge_response.strip():
                logger.warning(f"LLM Judge for {program_to_judge.id} returned an empty response.")
                return None, "LLM Judge returned empty response."

            logger.debug(f"Raw response from Judge LLM for {program_to_judge.id}: {raw_judge_response}")

            # Parse the JSON response
            judge_output = json.loads(raw_judge_response)
            score = judge_output.get("overall_score")
            justification = judge_output.get("justification")

            if score is None or justification is None:
                logger.warning(
                    f"LLM Judge response for {program_to_judge.id} missing 'overall_score' or 'justification'. Raw: {raw_judge_response}")
                return None, f"LLM Judge response malformed. Raw: {raw_judge_response[:200]}"

            logger.info(f"LLM Judge for {program_to_judge.id}: Score={score}, Justification='{justification[:100]}...'")
            return float(score), str(justification)

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from LLM Judge for {program_to_judge.id}: {e}. Raw response: {raw_judge_response}",
                exc_info=True)
            return None, f"LLM Judge JSONDecodeError. Raw: {raw_judge_response[:200]}"
        except Exception as e_judge:
            logger.error(f"Error during LLM Judge invocation for {program_to_judge.id}: {e_judge}", exc_info=True)
            return None, f"LLM Judge invocation error: {type(e_judge).__name__}"

    # --- MODIFIED: evaluate_program (Integrate LLM-as-Judge call) ---
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.debug(f"[evaluate_program] Starting for Program ID: {program.id}, Task ID: {task.id}")
        program.status = "evaluating"
        program.errors = [] # Clear previous errors
        program.llm_judge_feedback = None # Clear previous judge feedback

        default_scores = {
            "correctness": 0.0, "runtime_ms": float('inf'),
            "ruff_violations": float('inf'), "cyclomatic_complexity_avg": float('inf'),
            "maintainability_index": settings.DEFAULT_METRIC_VALUE.get("maintainability_index", -1.0),
            "passed_tests": 0.0,
            "total_tests": float(len(task.input_output_examples) if task.input_output_examples else 0),
            "runs_without_error": False,
            "standalone_script_runtime_ms": float('inf'),
            "llm_judge_overall_score": settings.DEFAULT_METRIC_VALUE.get("llm_judge_overall_score", 0.0) # NEW
        }
        program.fitness_scores = {**default_scores, **(program.fitness_scores if program.fitness_scores else {})}

        # 1. Syntax Check
        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.status = "failed_evaluation_syntax"
            logger.warning(f"[evaluate_program] Syntax errors for {program.id}: {syntax_errors}")
            # Ensure all relevant scores reflect failure
            program.fitness_scores["runs_without_error"] = False
            program.fitness_scores["ruff_violations"] = float('inf') # Can't run Ruff if syntax error
            return program
        logger.debug(f"[evaluate_program] Syntax check passed for {program.id}.")

        temp_code_file_path = None # Initialize
        try:
            # 2. Static Analysis
            if program.code.strip():
                # Create temp file (ensure this logic is robust as in your full version)
                _temp_file_obj = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8')
                _temp_file_obj.write(program.code)
                temp_code_file_path = _temp_file_obj.name
                _temp_file_obj.close()

                current_file_dir = os.path.dirname(os.path.abspath(__file__)) # Correct pathing for Ruff
                project_root_path = os.path.dirname(os.path.dirname(current_file_dir)) # Assuming evaluator_agent is in a subfolder of project root

                # --- Ruff Analysis (Ensuring all violations counted and logged) ---
                logger.info(f"Running Ruff analysis for program {program.id}...")
                try:
                    ruff_violations_list, ruff_violations_count_from_func = await asyncio.to_thread(
                        run_ruff_in_thread, temp_code_file_path, project_root_path
                    )
                    program.fitness_scores["ruff_violations"] = float(ruff_violations_count_from_func)

                    if ruff_violations_count_from_func > 0:
                        logger.info(f"Ruff found {ruff_violations_count_from_func} issues for program {program.id}.")
                        for i, violation in enumerate(ruff_violations_list):  # Iterate ALL
                            err_msg = (f"Ruff-{violation.get('code', 'UNK')}: "
                                       f"{violation.get('message', 'No message')} "
                                       f"@ L{violation.get('location', {}).get('row', 0)}, "
                                       f"C{violation.get('location', {}).get('column', 0)}")
                            program.errors.append(err_msg)
                            if i < 5: logger.debug(f"  - {err_msg}")
                            if i == 4 and ruff_violations_count_from_func > 5:
                                logger.debug(f"  ...and {ruff_violations_count_from_func - 5} more Ruff issues.")
                    else:
                        logger.info(f"Ruff analysis clean for program {program.id}.")
                        program.fitness_scores["ruff_violations"] = 0.0
                except Exception as e_ruff_call:
                    logger.error(f"Error calling/processing Ruff for {program.id}: {e_ruff_call}", exc_info=True)
                    program.errors.append(f"Ruff: Error during analysis - {type(e_ruff_call).__name__}")
                    program.fitness_scores["ruff_violations"] = float('inf')

                # --- Radon Analysis (Cyclomatic Complexity & Maintainability Index) ---
                needs_radon_cc = "cyclomatic_complexity_avg" in (task.primary_focus_metrics or []) or \
                                 task.improvement_mode == "general_refinement"
                needs_radon_mi = "maintainability_index" in (task.primary_focus_metrics or []) or \
                                 task.improvement_mode == "general_refinement"

                if needs_radon_cc:
                    logger.info(f"Running Radon CC analysis for program {program.id}...")
                    radon_cc_cmd = [sys.executable, "-m", "radon", "cc", "-j", temp_code_file_path]
                    cc_data, cc_error = await self._run_static_analysis_tool("Radon CC", radon_cc_cmd, program.id)
                    if cc_error:
                        program.errors.append(cc_error)
                        program.fitness_scores["cyclomatic_complexity_avg"] = float('inf')
                    elif cc_data and isinstance(cc_data, dict):
                        file_metrics = cc_data.get(os.path.basename(temp_code_file_path))
                        if not file_metrics and len(cc_data) == 1:
                            file_metrics = list(cc_data.values())[0]

                        if isinstance(file_metrics, list):
                            total_complexity = 0
                            count = 0
                            for item_metric in file_metrics:
                                if isinstance(item_metric, dict) and 'complexity' in item_metric:
                                    total_complexity += item_metric['complexity']
                                    count += 1
                            if count > 0:
                                program.fitness_scores["cyclomatic_complexity_avg"] = total_complexity / count
                                logger.info(
                                    f"Radon Avg CC for program {program.id}: {program.fitness_scores['cyclomatic_complexity_avg']:.2f}")
                            else:
                                program.fitness_scores["cyclomatic_complexity_avg"] = 0.0
                                logger.info(
                                    f"Radon Avg CC for program {program.id}: 0.0 (no complex blocks or empty file)")
                        else:
                            logger.warning(
                                f"Could not determine Radon CC from parsed JSON for {program.id}. Data: {cc_data}")
                            program.fitness_scores["cyclomatic_complexity_avg"] = float('inf')
                    else:  # cc_data is None or not a dict
                        program.fitness_scores["cyclomatic_complexity_avg"] = float(
                            'inf')  # Default if tool fails or no output

                if needs_radon_mi:
                    logger.info(f"Running Radon MI analysis for program {program.id}...")
                    radon_mi_cmd = [sys.executable, "-m", "radon", "mi", "-j", temp_code_file_path]
                    mi_data, mi_error = await self._run_static_analysis_tool("Radon MI", radon_mi_cmd, program.id)
                    if mi_error:
                        program.errors.append(mi_error)
                        program.fitness_scores["maintainability_index"] = settings.DEFAULT_METRIC_VALUE.get(
                            "maintainability_index", -1.0)
                    elif mi_data and isinstance(mi_data, dict):
                        file_metrics = mi_data.get(os.path.basename(temp_code_file_path))
                        if not file_metrics and len(mi_data) == 1:
                            file_metrics = list(mi_data.values())[0]

                        if isinstance(file_metrics, dict) and "mi" in file_metrics:
                            program.fitness_scores["maintainability_index"] = float(file_metrics["mi"])
                            logger.info(
                                f"Radon MI for program {program.id}: {program.fitness_scores['maintainability_index']:.2f}")
                        else:
                            logger.warning(
                                f"Could not determine Radon MI from parsed JSON for {program.id}. Data: {mi_data}")
                            program.fitness_scores["maintainability_index"] = settings.DEFAULT_METRIC_VALUE.get(
                                "maintainability_index", -1.0)
                    else:  # mi_data is None or not a dict
                        program.fitness_scores["maintainability_index"] = settings.DEFAULT_METRIC_VALUE.get(
                            "maintainability_index", -1.0)

            elif not program.code.strip():
                logger.info(f"[evaluate_program] Program {program.id} has empty code. Defaulting static scores.")
                program.fitness_scores["ruff_violations"] = 0.0
                program.fitness_scores["cyclomatic_complexity_avg"] = 0.0
                program.fitness_scores["maintainability_index"] = 100.0

            # 3. Standalone Script Execution Check (NEW - Blueprint Step 2)
            # Heuristic: If target_solution_description exists and implies a script, try to run it.
            # A more robust way would be a dedicated flag in TaskDefinition, e.g., `is_executable_script: bool`.
            # For now, let's assume we attempt this if `target_solution_description` is present.
            should_attempt_standalone_run = bool(task.target_solution_description) and program.code.strip()

            if should_attempt_standalone_run:
                logger.info(f"[evaluate_program] Attempting standalone execution for {program.id} based on task configuration.")
                runs_ok, script_stdout, script_stderr, script_exec_time_ms = await self._execute_standalone_script(
                    program_id=program.id,
                    code=program.code,
                    timeout_seconds=self.evaluation_timeout_seconds # Or a specific timeout for this step
                )
                program.fitness_scores["runs_without_error"] = runs_ok
                if script_exec_time_ms is not None:
                    program.fitness_scores["standalone_script_runtime_ms"] = script_exec_time_ms

                if not runs_ok:
                    program.status = "failed_evaluation_execution" # Set status if it crashes
                    error_log_msg = f"Standalone script execution failed for {program.id}."
                    if script_stderr: error_log_msg += f"\nSTDERR:\n{script_stderr}"
                    program.errors.append(error_log_msg)
                    logger.warning(f"[evaluate_program] {error_log_msg.splitlines()[0]}")
                else:
                    logger.info(f"[evaluate_program] Standalone script {program.id} executed successfully.")
                    if script_stdout: program.errors.append(f"Execution Log (STDOUT for {program.id}):\n{script_stdout}") # Log stdout for LLM judge
                    if script_stderr: program.errors.append(f"Execution Log (STDERR for {program.id} - non-fatal):\n{script_stderr}") # Log stderr even if RC 0

            # 4. Functional Evaluation
            # This section should populate program.fitness_scores["correctness"], ["passed_tests"], ["runtime_ms"]
            # and add execution-related errors to program.errors
            # The _execute_code_safely and _assess_correctness methods from your file seemed okay.
            if task.input_output_examples:
                logger.debug(f"[evaluate_program] Starting functional evaluation for {program.id}...")
                execution_results, execution_error_msg = await self._execute_code_safely(program.code,
                                                                                         task_for_examples=task)
                if execution_error_msg:
                    program.errors.append(f"Execution Error: {execution_error_msg}")
                    program.fitness_scores["correctness"] = 0.0
                elif execution_results and "test_outputs" in execution_results:
                    correctness, passed_tests, total_tests = self._assess_correctness(execution_results,
                                                                                      task.input_output_examples)
                    program.fitness_scores["correctness"] = correctness
                    program.fitness_scores["passed_tests"] = float(passed_tests)
                    avg_runtime = execution_results.get("average_runtime_ms")
                    if avg_runtime is not None: program.fitness_scores["runtime_ms"] = avg_runtime
                    if correctness < 1.0:  # Add summary of test failures if not all passed
                        program.errors.append(
                            f"Failed {int(total_tests - passed_tests)} out of {int(total_tests)} I/O test cases.")
                else:  # Malformed execution results
                    program.errors.append("Execution Error: Unknown issue or malformed results from sandbox.")
                    program.fitness_scores["correctness"] = 0.0
            elif task.improvement_mode == "task_focused" and not task.input_output_examples:  # No tests to run
                program.errors.append("Evaluation: Task 'task_focused' but no I/O examples for correctness check.")
                # Correctness remains default 0, or could be set to 1.0 if no tests means "not failed"
                # Let's assume correctness 0 if it can't be verified for task_focused.

            # --- 5. LLM-as-a-Judge Invocation (NEW - Blueprint Step 3) ---
            if program.status not in ["failed_evaluation_syntax"]: # Don't judge if basic syntax error
                # Populate the execution_summary_for_judge with final values from program.fitness_scores and program.errors
                # This summary is what PromptDesignerAgent will use.
                # This needs to be done carefully to pass the right info.
                # For now, let's assume execution_summary_for_judge is built up correctly in prior steps.
                # Example:
                current_exec_summary = {
                    "runs_without_error": program.fitness_scores.get("runs_without_error"),
                    "standalone_script_runtime_ms": program.fitness_scores.get("standalone_script_runtime_ms"),
                    "stdout": next((e.split("STDOUT for")[1].strip() for e in program.errors if "STDOUT for" in e), ""), # Simple extraction
                    "stderr": next((e.split("STDERR for")[1].strip() for e in program.errors if "STDERR for" in e or "Standalone script execution failed" in e), ""), # Simple extraction
                    "ruff_violations": program.fitness_scores.get("ruff_violations"),
                    "correctness": program.fitness_scores.get("correctness")
                }

                judge_score, judge_justification = await self._invoke_llm_judge(
                    task, program, current_exec_summary
                )
                if judge_score is not None:
                    program.fitness_scores["llm_judge_overall_score"] = judge_score
                if judge_justification is not None:
                    program.llm_judge_feedback = judge_justification
                if judge_score is None and judge_justification: # If scoring failed but got some feedback
                    program.errors.append(f"LLM Judge Note: {judge_justification}")

            # --- UNTANGLED AND REFINED Final Status Determination Logic ---
            if program.status == "evaluating":  # Check if status hasn't been set to a critical failure already

                # Identify critical errors (errors not from Ruff, and not simple I/O test failure summaries)
                # The "Failed X out of Y test cases" is informational for the LLM, not a "crash" error.
                critical_error_messages = [
                    e for e in program.errors
                    if not e.strip().lower().startswith("ruff-") and
                       not e.strip().lower().startswith("failed")  # Exclude "Failed X out of Y I/O test cases"
                ]

                current_correctness = program.fitness_scores.get("correctness", 0.0)

                if critical_error_messages:  # Any execution crash, sandbox issue, etc.
                    program.status = "failed_evaluation_execution"
                elif task.input_output_examples and current_correctness < 1.0:  # Failed functional tests
                    program.status = "failed_evaluation_tests"
                else:  # No critical errors, and 100% correct (or no I/O tests for non-task_focused modes)
                    program.status = "evaluated"
                    # At this point, program.errors might still contain Ruff messages.
                    # This means "evaluated with lint issues", which is fine.

            # One final check: if no I/O examples were provided and it's task_focused, status might need adjustment
            # if we didn't set it to a failure above.
            # If mode is task_focused, no I/O examples provided, and no other critical errors -> it's hard to judge.
            # For now, the logic above would set it to "evaluated" if no critical errors.
            # The `program.errors.append("Evaluation: Task 'task_focused' but no I/O examples for correctness check.")`
            # will be a hint.

        except Exception as e_evaluate_program:  # Catch-all for the entire try block
            logger.error(f"[evaluate_program] CRITICAL unhandled exception: {e_evaluate_program}", exc_info=True)
            program.errors.append(
                f"Critical Evaluation Error: {type(e_evaluate_program).__name__} - {str(e_evaluate_program)}")
            program.status = "failed_evaluation_internal_critical"
            for key in program.fitness_scores:
                if key in ["ruff_violations", "runtime_ms", "cyclomatic_complexity_avg"]:
                    program.fitness_scores[key] = float('inf')
                elif key in ["correctness", "passed_tests", "maintainability_index"]:
                    program.fitness_scores[key] = 0.0 if key != "maintainability_index" else -1.0
        finally:
            if temp_code_file_path and os.path.exists(temp_code_file_path):
                try:
                    os.remove(temp_code_file_path)
                except Exception as e_cleanup:
                    logger.error(f"Error cleaning temp file {temp_code_file_path}: {e_cleanup}")

        logger.info(
            f"[evaluate_program] FINALIZED for Program ID: {program.id}. Status: {program.status}. "
            f"Runs w/o Error: {program.fitness_scores.get('runs_without_error', 'N/A')}, "
            f"Correctness (I/O): {program.fitness_scores.get('correctness', 0.0) * 100:.1f}%, "
            f"Ruff Violations: {program.fitness_scores.get('ruff_violations', 'N/A')}, "
            f"Errors logged: {len(program.errors)}"
        )
        # logger.debug(f"Full fitness scores for {program.id}: {program.fitness_scores}")
        if program.errors: logger.debug(f"Detailed errors/logs for {program.id}: {program.errors}")
        return program

    async def execute(self, *args, **kwargs) -> Program:  # Signature changed to match BaseAgent
        """
        Generic execute method required by BaseAgent.
        This implementation expects 'program' and 'task' in kwargs
        and delegates to self.evaluate_program.
        """
        program_arg = kwargs.get("program")
        task_arg = kwargs.get("task")
        logger.debug(
            f"[execute wrapper] Called with program: {program_arg.id if program_arg else 'None'}, task: {task_arg.id if task_arg else 'None'}")

        if not isinstance(program_arg, Program):
            err_msg = f"EvaluatorAgent.execute expects 'program' of type Program in kwargs, but got {type(program_arg)}."
            logger.error(err_msg)
            raise TypeError(err_msg)

        if not isinstance(task_arg, TaskDefinition):
            err_msg = f"EvaluatorAgent.execute expects 'task' of type TaskDefinition in kwargs, but got {type(task_arg)}."
            logger.error(err_msg)
            raise TypeError(err_msg)

        logger.debug(
            f"EvaluatorAgent.execute is delegating to evaluate_program for program '{program_arg.id}', task '{task_arg.id}'")
        return await self.evaluate_program(program_arg, task_arg)