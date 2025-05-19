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
from pylint import lint # Main Pylint entry point for Run
from pylint.lint import PyLinter # For more direct control
from pylint.reporters.text import TextReporter # We'll need a reporter
from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition # BaseAgent is inherited via EvaluatorAgentInterface
from config import settings

logger = logging.getLogger(__name__)


# Method_v1.3.7 ("Trust Pylint with RCFile" Attempt)
def run_pylint_in_thread(file_path: str) -> tuple[Optional[float], int]:
    temp_file_directory = os.path.dirname(file_path)
    original_sys_path = list(sys.path)

    current_script_path = os.path.abspath(__file__)
    evaluator_agent_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(evaluator_agent_dir)
    pylintrc_path = os.path.join(project_root, ".pylintrc")

    logger.debug(f"run_pylint_in_thread: Determined absolute path for .pylintrc: {pylintrc_path}")

    # --- Crucial Pylint Options ---
    # These are options that configure Pylint's behavior or are essential for our use.
    pylint_opts = [
        file_path,  # The file to lint MUST come AFTER options for lint.Run
        "--persistent=n",  # Disable caching of results
        "--score=y",  # We definitely want the score output
    ]

    if os.path.exists(pylintrc_path):
        pylint_opts.append(f"--rcfile={pylintrc_path}")
        logger.info(f"Using .pylintrc: {pylintrc_path}")
    else:
        logger.warning(f"No .pylintrc found at {pylintrc_path}. Pylint will use defaults.")
        # If no rcfile, we might need some basic default enables if Pylint doesn't run much by default.
        # However, Pylint's default is to enable most things. Let's see.
        # We could add: pylint_opts.append("--enable=C,R,W,E,F") here if needed.

    # This was our specific fix for an overgeneral exceptions warning pattern.
    # It's a direct command-line argument that Pylint understands.
    pylint_opts.append("--overgeneral-exceptions=")

    logger.debug(f"run_pylint_in_thread: Effective pylint_opts for lint.Run: {pylint_opts}")

    captured_pylint_stdout = io.StringIO()  # For Pylint's actual report
    captured_pylint_stderr = io.StringIO()  # For Pylint's own errors/warnings

    linter_instance_from_run = None
    score = None
    statements = 0

    try:
        # Temporarily modify sys.path for Pylint to find the temp file as a module
        if temp_file_directory not in sys.path:
            sys.path.insert(0, temp_file_directory)
            logger.debug(
                f"run_pylint_in_thread: Added '{temp_file_directory}' to sys.path for Pylint module resolution.")

        # Use redirect_stderr to catch Pylint's internal errors/warnings if it prints to stderr
        # The TextReporter (default for lint.Run unless output-format is changed) writes to stdout.
        with redirect_stdout(captured_pylint_stdout), redirect_stderr(captured_pylint_stderr):
            # lint.Run is the standard high-level way to invoke Pylint programmatically.
            # It handles linter creation, config loading (rcfile + cmdline), checking, and reporting.
            # The `exit=False` prevents it from calling sys.exit().
            pylint_run = lint.Run(pylint_opts, exit=False)

            # After Run completes, the linter instance it used should have stats
            if hasattr(pylint_run, 'linter') and pylint_run.linter is not None:
                linter_instance_from_run = pylint_run.linter
            else:
                logger.error("lint.Run did not produce a usable linter instance.")
                # No need to restore sys.path here, finally block handles it.
                return None, 0  # Cannot get score if linter is missing

        # --- End of Pylint execution block ---

        pylint_report_output = captured_pylint_stdout.getvalue()
        pylint_error_output = captured_pylint_stderr.getvalue()

        if pylint_report_output.strip():
            logger.debug(f"Pylint captured stdout (report) for {file_path}:\n{pylint_report_output}")
        if pylint_error_output.strip():
            # This is where we'd see the AttributeError if it still happens, or other Pylint errors.
            logger.warning(f"Pylint captured stderr for {file_path}:\n{pylint_error_output}")

        # Check stderr for the specific AttributeError or other critical errors
        if "AttributeError: 'Namespace' object has no attribute 'mixin_class_rgx'" in pylint_error_output:
            logger.critical(f"PERSISTENT AttributeError for mixin_class_rgx with file {file_path}!")
            # This would indicate a very fundamental issue with how this Pylint version handles this rcfile option.
        elif "fatal" in pylint_error_output.lower() or "astroid-error" in pylint_error_output.lower():
            logger.error(f"Pylint reported a fatal/astroid error for {file_path}. Score might be unreliable.")
            # Fall through to try and get score, but it will likely be 0 or low.

        # Extract score and statements from the linter instance used by Run
        if linter_instance_from_run and hasattr(linter_instance_from_run,
                                                'stats') and linter_instance_from_run.stats is not None:
            if hasattr(linter_instance_from_run.stats, 'global_note'):
                score = linter_instance_from_run.stats.global_note
            if hasattr(linter_instance_from_run.stats, 'statement'):
                statements = linter_instance_from_run.stats.statement

            if score is None and not (
                    "fatal" in pylint_error_output.lower() or "astroid-error" in pylint_error_output.lower()):
                # If score is None but no obvious Pylint crash, log details from stats
                s = linter_instance_from_run.stats
                logger.warning(
                    f"Pylint global_note (score) is None for {file_path}. Stats: E={getattr(s, 'error', 'N/A')}, "
                    f"W={getattr(s, 'warning', 'N/A')}, C={getattr(s, 'convention', 'N/A')}, R={getattr(s, 'refactor', 'N/A')}"
                )
        elif score is None:  # Catch all if score is still None (e.g., due to earlier error or no stats)
            logger.warning(
                f"Pylint score is None for {file_path}. Check Pylint stderr for errors. Setting score to 0.0 if error occurred.")
            if "fatal" in pylint_error_output.lower() or "astroid-error" in pylint_error_output.lower() or "AttributeError" in pylint_error_output:
                score = 0.0

        logger.debug(f"Pylint run on {file_path} completed. Score: {score}, Statements: {statements}")
        return score, statements

    except Exception as e_pylint_thread:  # Catch any other unexpected exception during this process
        logger.error(
            f"CRITICAL Exception in run_pylint_in_thread for file {file_path}: {e_pylint_thread}",
            exc_info=True
        )
        # Log any captured output even on broader exception
        stdout_on_exc = captured_pylint_stdout.getvalue()
        stderr_on_exc = captured_pylint_stderr.getvalue()
        if stdout_on_exc.strip(): logger.debug(f"Pylint stdout on broader exception: {stdout_on_exc}")
        if stderr_on_exc.strip(): logger.error(f"Pylint stderr on broader exception: {stderr_on_exc}")
        return None, 0
    finally:
        # Crucial: Restore sys.path
        sys.path = original_sys_path
        logger.debug(f"run_pylint_in_thread: Restored original sys.path in finally block ({temp_file_directory}).")


# To this (BaseAgent is inherited via EvaluatorAgentInterface):
class EvaluatorAgent(EvaluatorAgentInterface):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        # super().__init__() will correctly call BaseAgent.__init__ via the MRO
        super().__init__(config=None) # Pass config=None, as EvaluatorAgent doesn't seem to use its own specific config dict
        self.task_definition = task_definition
        self.evaluation_model_name = settings.EVALUATION_MODEL_NAME
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        logger.info(f"EvaluatorAgent initialized with model: {self.evaluation_model_name}, timeout: {self.evaluation_timeout_seconds}s")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        logger.debug(f"EvaluatorAgent instance created. Task ID: {self.task_definition.id if self.task_definition else 'None'}")

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

    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.debug(
            f"[evaluate_program] Starting for Program ID: {program.id}, Task ID: {task.id}. Current Status: {program.status}")
        logger.debug(
            f"[evaluate_program] Program Code (first 200 chars):\n{program.code[:200].replace(chr(10), '<NL>')}")

        program.status = "evaluating"  # Set status at the beginning
        program.errors = []  # Clear previous errors
        # Initialize fitness_scores if not already, or ensure keys exist
        default_scores = {
            "correctness": 0.0, "runtime_ms": float('inf'),
            "pylint_score": settings.DEFAULT_METRIC_VALUE.get("pylint_score", -10.0),
            "cyclomatic_complexity_avg": float('inf'),
            "maintainability_index": settings.DEFAULT_METRIC_VALUE.get("maintainability_index", -1.0),
            "passed_tests": 0.0,
            "total_tests": float(len(task.input_output_examples) if task.input_output_examples else 0)
        }
        program.fitness_scores = {**default_scores, **program.fitness_scores}

        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.status = "failed_evaluation_syntax"  # More specific status
            logger.warning(f"[evaluate_program] Syntax errors for {program.id}: {syntax_errors}")
            logger.debug(
                f"[evaluate_program] FINALIZING for {program.id}. Status: {program.status}. Fitness: {program.fitness_scores}. Errors: {program.errors}. Returning program object.")
            return program  # Return the program object with errors
        logger.debug(f"[evaluate_program] Syntax check passed for {program.id}.")

        # --- Static Analysis Block ---
        temp_code_file_path = None
        if program.code.strip():
            logger.debug(f"[evaluate_program] Proceeding with static analysis for {program.id}")
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmpf:
                    tmpf.write(program.code)
                    temp_code_file_path = tmpf.name
                logger.debug(f"[evaluate_program] Temp code file for static analysis: {temp_code_file_path}")

                needs_pylint = "pylint_score" in (task.primary_focus_metrics or []) or \
                               task.improvement_mode == "general_refinement"

                if needs_pylint:
                    logger.info(f"Running Pylint analysis (as library) for program {program.id}...")
                    try:
                        # REMOVE THE NESTED DEFINITION OF run_pylint_in_thread HERE! ✨
                        # No more: def run_pylint_in_thread(file_path): ...

                        # This call will now use the top-level/module-level run_pylint_in_thread
                        # which contains our fixes (Method_v1.2.2).
                        pylint_score_val, pylint_statement_count = await asyncio.to_thread(
                            run_pylint_in_thread, # Correctly refers to the outer/fixed version
                            temp_code_file_path
                        )

                        if pylint_statement_count == 0 and not program.code.strip().startswith("def"):
                            # If no statements were found by Pylint (e.g. empty file, or just comments)
                            # and it's not clearly just a function definition (which might have 0 statements if it's just `pass`)
                            # assign a neutral or low score.
                            logger.info(
                                f"Pylint found 0 statements for program {program.id}. Assigning Pylint score of 0.0.")
                            program.fitness_scores["pylint_score"] = 0.0
                        elif pylint_score_val is not None:
                            program.fitness_scores["pylint_score"] = float(pylint_score_val)
                            logger.info(
                                f"Pylint score for program {program.id}: {program.fitness_scores['pylint_score']:.2f}/10")
                        else:
                            logger.warning(
                                f"Could not retrieve Pylint score for {program.id} using library method. linter.stats.global_note was None.")
                            program.errors.append("Pylint: Failed to retrieve score programmatically.")
                            # Keep the default -10.0 to indicate an issue
                    except Exception as e_pylint:
                        logger.error(f"Error running Pylint as library for {program.id}: {e_pylint}", exc_info=True)
                        program.errors.append(f"Pylint: Error during analysis - {type(e_pylint).__name__}")
                        # Keep the default -10.0

                # --- Radon Analysis (remains the same, uses subprocess) ---
                needs_radon_cc = "cyclomatic_complexity_avg" in (task.primary_focus_metrics or []) or \
                                 task.improvement_mode == "general_refinement"
                needs_radon_mi = "maintainability_index" in (task.primary_focus_metrics or []) or \
                                 task.improvement_mode == "general_refinement"

                # Radon Analysis - Cyclomatic Complexity (using JSON output)
                if needs_radon_cc:
                    logger.info(f"Running Radon CC analysis for program {program.id}...")
                    radon_cc_cmd = [sys.executable, "-m", "radon", "cc", "-j", temp_code_file_path]
                    cc_data, cc_error = await self._run_static_analysis_tool("Radon CC", radon_cc_cmd, program.id)
                    if cc_error:
                        program.errors.append(cc_error)
                    elif cc_data and isinstance(cc_data, dict):  # Expecting a dict keyed by filename
                        # The JSON output is a dict where keys are filenames.
                        # We expect only one file, our temp_code_file_path.
                        file_metrics = cc_data.get(
                            os.path.basename(temp_code_file_path))  # Or use the full path if radon returns that
                        if not file_metrics and len(cc_data) == 1:  # If only one key, assume it's our file
                            file_metrics = list(cc_data.values())[0]

                        if isinstance(file_metrics,
                                      list):  # Radon cc -j output is a list of dicts for functions/classes
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
                            else:  # Empty file or no analysable blocks
                                program.fitness_scores["cyclomatic_complexity_avg"] = 0.0
                                logger.info(
                                    f"Radon Avg CC for program {program.id}: 0.0 (no complex blocks found or empty file)")
                        else:  # Fallback if parsing structure is unexpected or for empty files that might give empty dict.
                            logger.warning(
                                f"Could not determine Radon CC from parsed JSON for {program.id}. Data: {cc_data}")
                            program.fitness_scores["cyclomatic_complexity_avg"] = float('inf')  # Indicate error/unknown

                # Radon Analysis - Maintainability Index (using JSON output)
                if needs_radon_mi:
                    logger.info(f"Running Radon MI analysis for program {program.id}...")
                    radon_mi_cmd = [sys.executable, "-m", "radon", "mi", "-j", temp_code_file_path]
                    mi_data, mi_error = await self._run_static_analysis_tool("Radon MI", radon_mi_cmd, program.id) # No temp_file_path here
                    if mi_error:
                        program.errors.append(mi_error)
                    elif mi_data and isinstance(mi_data, dict):
                        file_metrics = mi_data.get(os.path.basename(temp_code_file_path))
                        if not file_metrics and len(mi_data) == 1:
                            file_metrics = list(mi_data.values())[0]

                        if isinstance(file_metrics, dict) and "mi" in file_metrics:
                            program.fitness_scores["maintainability_index"] = float(file_metrics["mi"])
                            logger.info(
                                f"Radon MI for program {program.id}: {program.fitness_scores['maintainability_index']:.2f}")
                        else:  # Fallback for empty files or unexpected structure.
                            logger.warning(
                                f"Could not determine Radon MI from parsed JSON for {program.id}. Data: {mi_data}")
                            program.fitness_scores["maintainability_index"] = -1.0  # Indicate error/unknown

            except Exception as e_static:
                logger.error(
                    f"[evaluate_program] Unexpected error during static analysis setup/exec for {program.id}: {e_static}",
                    exc_info=True)
                program.errors.append(f"Static Analysis: General error - {type(e_static).__name__}")
                # Don't return yet, try functional evaluation if possible, or let it proceed to set status.
            finally:
                if temp_code_file_path and os.path.exists(temp_code_file_path):
                    os.remove(temp_code_file_path)
                    logger.debug(f"[evaluate_program] Removed temp static analysis file: {temp_code_file_path}")
        elif not program.code.strip():
            logger.info(
                f"[evaluate_program] Program {program.id} has empty code. Skipping static analysis, assigning default 'bad' static scores.")
            # Defaults already set, but can re-affirm or log here.

        # --- Functional Evaluation (I/O examples) ---
        if task.input_output_examples:
            logger.debug(
                f"[evaluate_program] Starting functional evaluation for {program.id} against {len(task.input_output_examples)} test cases.")
            execution_results, execution_error = await self._execute_code_safely(program.code, task_for_examples=task)

            logger.debug(
                f"[evaluate_program] _execute_code_safely returned for {program.id} -> execution_results: {str(execution_results)[:200]}..., execution_error: {execution_error}")

            if execution_error:
                logger.warning(f"[evaluate_program] Execution error for {program.id}: {execution_error}")
                program.errors.append(f"Execution Error: {execution_error}")
                program.fitness_scores["correctness"] = 0.0
                program.status = "failed_evaluation_execution"  # More specific
            elif execution_results and "test_outputs" in execution_results:  # Ensure "test_outputs" exists
                logger.debug(
                    f"[evaluate_program] Execution results for {program.id} (first 200 chars of test_outputs): {str(execution_results.get('test_outputs'))[:200]}")
                correctness, passed_tests, total_tests = self._assess_correctness(execution_results,
                                                                                  task.input_output_examples)
                logger.debug(
                    f"[evaluate_program] _assess_correctness for {program.id} -> correctness: {correctness}, passed: {passed_tests}, total: {total_tests}")
                program.fitness_scores["correctness"] = correctness
                program.fitness_scores["passed_tests"] = float(passed_tests)
                # program.fitness_scores["total_tests"] already set from default_scores
                if "average_runtime_ms" in execution_results and correctness > 0:
                    program.fitness_scores["runtime_ms"] = execution_results["average_runtime_ms"]

                if correctness < 1.0 and not execution_error:  # Don't double-log if execution_error already captured it
                    program.errors.append(f"Failed {total_tests - passed_tests} out of {total_tests} I/O test cases.")
                    if not program.status == "evaluating":  # If not already set to a more specific failure
                        program.status = "failed_evaluation_tests"
            else:  # execution_results is None or malformed, and no execution_error string
                logger.warning(
                    f"[evaluate_program] Execution of {program.id} yielded no results or malformed results, and no specific error message from _execute_code_safely. Raw results: {str(execution_results)[:200]}")
                program.errors.append("Execution Error: Unknown issue or malformed results from sandbox.")
                program.fitness_scores["correctness"] = 0.0
                program.status = "failed_evaluation_unknown_exec"

        elif task.improvement_mode == "task_focused" and not task.input_output_examples:
            logger.warning(
                f"[evaluate_program] Task {task.id} is 'task_focused' but has no I/O examples. Cannot assess correctness.")
            program.errors.append("Evaluation: Task 'task_focused' but no I/O examples.")
            # Status remains 'evaluating' or 'failed_evaluation_syntax' if that happened.
            # If no errors so far, it's technically evaluated but without correctness.
            if not program.errors: program.status = "evaluated_no_tests"

        # Final status determination
        if not program.errors and program.status == "evaluating":  # If it made it through without errors
            program.status = "evaluated"
        elif program.errors and program.status == "evaluating":  # If errors were added but status not specifically set to a failure type
            program.status = "failed_evaluation_generic"

        logger.debug(
            f"[evaluate_program] FINALIZING for {program.id}. Status: {program.status}. Fitness: {program.fitness_scores}. Errors: {program.errors}. About to return program object.")
        return program

    # MODIFIED execute method:
    async def execute(self, *args, **kwargs) -> Program:  # Signature changed to match BaseAgent
        """
        Generic execute method required by BaseAgent.
        This implementation expects 'program' and 'task' in kwargs
        and delegates to self.evaluate_program.
        """
        program_arg = kwargs.get("program")
        task_arg = kwargs.get("task")
        logger.debug(f"[execute wrapper] Called with program: {program_arg.id if program_arg else 'None'}, task: {task_arg.id if task_arg else 'None'}")

        # Validate that we got the arguments we need to call evaluate_program
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
        # Now, call the main evaluation logic
        return await self.evaluate_program(program_arg, task_arg)