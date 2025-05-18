import time
import logging
import traceback
import subprocess
import tempfile
import os
import ast
import json
import asyncio
import sys
from typing import Optional, Dict, Any, Tuple, Union, List
import re # Already here for Pylint

from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = settings.GEMINI_EVALUATION_MODEL
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        logger.info(f"EvaluatorAgent initialized with model: {self.evaluation_model_name}, timeout: {self.evaluation_timeout_seconds}s")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")

    def _check_syntax(self, code: str) -> List[str]: # Unchanged
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}")
        except Exception as e:
            errors.append(f"Unexpected error during syntax check: {str(e)}")
        return errors

    async def _execute_code_safely(
        self,
        code: str,
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        results = {"test_outputs": [], "average_runtime_ms": 0.0}
        
        if not task_for_examples.input_output_examples:
            logger.warning("No input/output examples provided to _execute_code_safely.")
            return results, "No test cases to run."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task {task_for_examples.id} does not specify 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        def serialize_arg(arg):
            if isinstance(arg, (float, int)) and (arg == float('inf') or arg == float('-inf') or arg != arg):
                return f"float('{str(arg)}')"
            return json.dumps(arg)

        # Convert input_output_examples to a string with proper Python values for Infinity
        test_cases_str = json.dumps(task_for_examples.input_output_examples)
        test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")')
        test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')

        test_harness_code = f"""
import json
import time
import sys
import math  # Import math for inf/nan constants

# User's code (function to be tested)
{code}

# Test execution logic
results = []
total_execution_time = 0
num_tests = 0

# Special constants for test cases
Infinity = float('inf')
NaN = float('nan')

test_cases = {test_cases_str} 
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

for i, test_case in enumerate(test_cases):
    input_args = test_case.get("input")
    
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
        with open(temp_file_path, "w") as f:
            f.write(test_harness_code)

        cmd = [sys.executable, temp_file_path]
        
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

            if proc.returncode != 0:
                error_message = f"Execution failed with exit code {proc.returncode}. Stdout: '{stdout_str}'. Stderr: '{stderr_str}'"
                logger.warning(error_message)
                return None, error_message
            
            if not stdout_str:
                 logger.warning(f"Execution produced no stdout. Stderr: '{stderr_str}'")
                 return None, f"No output from script. Stderr: '{stderr_str}'"

            try:
                def json_loads_with_infinity(s):
                    s = s.replace('"Infinity"', 'float("inf")')
                    s = s.replace('"-Infinity"', 'float("-inf")')
                    s = s.replace('"NaN"', 'float("nan")')
                    return json.loads(s)

                parsed_output = json_loads_with_infinity(stdout_str)
                logger.debug(f"Parsed execution output: {parsed_output}")
                return parsed_output, None
            except json.JSONDecodeError as e:
                error_message = f"Failed to decode JSON output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
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
            logger.warning(f"Code execution timed out after {timeout} seconds for function {task_for_examples.function_name_to_evolve}.")
            return None, f"Execution timed out after {timeout} seconds."
        except Exception as e:
            logger.error(f"An unexpected error occurred during code execution: {e}", exc_info=True)
            return None, f"Unexpected execution error: {str(e)}"
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e_cleanup:
                logger.error(f"Error during cleanup of temp files: {e_cleanup}")

    def _assess_correctness(self, execution_results: Dict[str, Any], expected_outputs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        passed_tests = 0
        total_tests = len(expected_outputs)
        
        if not execution_results or "test_outputs" not in execution_results:
            logger.warning("Execution results are missing 'test_outputs' field.")
            return 0.0, 0, total_tests

        actual_test_outputs = execution_results["test_outputs"]

        if len(actual_test_outputs) != total_tests:
            logger.warning(f"Mismatch in number of test outputs ({len(actual_test_outputs)}) and expected outputs ({total_tests}). Some tests might have crashed before producing output.")
        
        for i, expected in enumerate(expected_outputs):
            actual_output_detail = next((res for res in actual_test_outputs if res.get("test_case_id") == i), None)

            if actual_output_detail and actual_output_detail.get("status") == "success":
                actual = actual_output_detail.get("output")
                expected_val = expected["output"]
                
                if actual == expected_val:
                    passed_tests += 1
                else:
                    logger.debug(f"Test case {i} failed: Expected '{expected_val}', Got '{actual}'")
            elif actual_output_detail:
                logger.debug(f"Test case {i} had error: {actual_output_detail.get('error')}")
            else:
                logger.debug(f"Test case {i}: No output found in results.")

        if total_tests == 0:
            return 1.0, 0, 0
        
        correctness = passed_tests / total_tests
        return correctness, passed_tests, total_tests

    async def _run_static_analysis_tool(self, tool_name: str, command: List[str], program_id: str,
                                        temp_file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Helper function to run a static analysis tool and parse its JSON output."""
        logger.debug(f"Executing {tool_name} command: {' '.join(command)}")
        try:
            process = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, universal_newlines=True, timeout=60
                # Changed text to universal_newlines_v6
            )
            stdout = process.stdout
            stderr = process.stderr

            if stderr and process.returncode != 0:  # Some tools might print warnings to stderr but still succeed
                logger.warning(f"{tool_name} stderr for {program_id} (return code {process.returncode}):\n{stderr}")
            if not stdout and process.returncode != 0:  # If no stdout and error, it's a problem
                logger.error(
                    f"{tool_name} for {program_id} produced no output and failed (code {process.returncode}). Stderr: {stderr}")
                return None, f"{tool_name}: No output and failed (code {process.returncode})."
            if not stdout:  # No output but success code, maybe an empty file or no issues found by some specific radon command.
                logger.info(
                    f"{tool_name} for {program_id} produced no stdout, but exited successfully. Assuming valid empty result.")
                return {}, None  # Return empty dict for successful no-output scenario

            logger.debug(f"{tool_name} stdout for {program_id}:\n{stdout[:1000]}...")

            # Attempt to parse JSON output
            try:
                parsed_json = json.loads(stdout)
                return parsed_json, None
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode {tool_name} JSON output for {program_id}: {e}. Raw output: '{stdout[:500]}'")
                return None, f"{tool_name}: Failed to decode JSON output."

        except subprocess.TimeoutExpired:
            logger.error(f"{tool_name} execution timed out for program {program_id}.")
            return None, f"{tool_name}: Execution timed out."
        except FileNotFoundError:
            logger.error(f"{tool_name} command not found. Make sure {tool_name} is installed and in PATH.")
            return None, f"{tool_name}: Command not found. Is {tool_name} installed?"
        except Exception as e:
            logger.error(f"Error during {tool_name} analysis for program {program_id}: {e}", exc_info=True)
            return None, f"{tool_name}: Analysis error - {type(e).__name__}"

    async def evaluate_program(self, program: Program,
                               task: TaskDefinition) -> Program:  # Method_v3 (Radon Integration)
        logger.info(f"Evaluating program: {program.id} for task: {task.id} (Mode: {task.improvement_mode})")
        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {
            "correctness": 0.0,
            "runtime_ms": float('inf'),
            "pylint_score": -1.0,
            "cyclomatic_complexity_avg": float('inf'),  # Initialize new Radon metric (lower is better) _v3
            "maintainability_index": -1.0  # Initialize new Radon metric (higher is better) _v3
        }

        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.status = "failed_evaluation"
            logger.warning(f"Syntax errors found in program {program.id}: {syntax_errors}")
            return program
        logger.debug(f"Syntax check passed for program {program.id}.")

        # --- Static Analysis Section (Pylint and Radon) ---
        # Only run static analysis if in general_refinement or if metrics are explicitly requested
        needs_pylint = "pylint_score" in (task.primary_focus_metrics or [])
        needs_radon_cc = "cyclomatic_complexity_avg" in (task.primary_focus_metrics or [])
        needs_radon_mi = "maintainability_index" in (task.primary_focus_metrics or [])

        run_static_analysis = task.improvement_mode == "general_refinement" or \
                              needs_pylint or needs_radon_cc or needs_radon_mi

        temp_code_file_path = None  # Define here for broader scope in finally block
        if run_static_analysis and program.code.strip():  # Only run on non-empty code
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmpf:
                    tmpf.write(program.code)
                    temp_code_file_path = tmpf.name

                # Pylint Analysis ( reusing structure from _run_static_analysis_tool )
                if task.improvement_mode == "general_refinement" or needs_pylint:
                    logger.info(f"Running Pylint analysis for program {program.id}...")
                    pylint_cmd = [sys.executable, "-m", "pylint", temp_code_file_path, "--output-format=text"]
                    # Note: --rcfile behavior might differ on Windows, consider removing or adjusting.
                    if sys.platform != "win32": pylint_cmd.insert(4, "--rcfile=/dev/null")

                    # Pylint doesn't have a straightforward JSON for just the score, parse text.
                    process_pylint = await asyncio.to_thread(
                        subprocess.run, pylint_cmd, capture_output=True, universal_newlines=True, timeout=60  # _v6
                    )
                    pylint_output = process_pylint.stdout
                    if process_pylint.stderr and process_pylint.returncode != 0:
                        logger.warning(f"Pylint stderr for {program.id}:\n{process_pylint.stderr}")

                    score_match = re.search(r"Your code has been rated at (-?\d+\.?\d*)/10", pylint_output)
                    if score_match:
                        program.fitness_scores["pylint_score"] = float(score_match.group(1))
                        logger.info(
                            f"Pylint score for program {program.id}: {program.fitness_scores['pylint_score']}/10")
                    else:
                        logger.warning(f"Could not parse Pylint score for {program.id}. Output: {pylint_output[:500]}")
                        program.errors.append("Pylint: Failed to parse score.")

                # Radon Analysis - Cyclomatic Complexity (using JSON output)
                if task.improvement_mode == "general_refinement" or needs_radon_cc:
                    logger.info(f"Running Radon CC analysis for program {program.id}...")
                    radon_cc_cmd = [sys.executable, "-m", "radon", "cc", "-j", temp_code_file_path]
                    cc_data, cc_error = await self._run_static_analysis_tool("Radon CC", radon_cc_cmd, program.id,
                                                                             temp_code_file_path)
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
                if task.improvement_mode == "general_refinement" or needs_radon_mi:
                    logger.info(f"Running Radon MI analysis for program {program.id}...")
                    radon_mi_cmd = [sys.executable, "-m", "radon", "mi", "-j", temp_code_file_path]
                    mi_data, mi_error = await self._run_static_analysis_tool("Radon MI", radon_mi_cmd, program.id,
                                                                             temp_code_file_path)
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

            except Exception as e_static:  # Catch-all for issues in the static analysis block
                logger.error(f"Unexpected error during static analysis setup or execution for {program.id}: {e_static}",
                             exc_info=True)
                program.errors.append(f"Static Analysis: General error - {type(e_static).__name__}")
            finally:
                if temp_code_file_path and os.path.exists(temp_code_file_path):
                    os.remove(temp_code_file_path)
        elif not program.code.strip():
            logger.info(f"Program {program.id} has empty code. Skipping static analysis.")
            # Assign neutral/worst scores for empty code to avoid issues in selection
            program.fitness_scores["pylint_score"] = 0.0
            program.fitness_scores["cyclomatic_complexity_avg"] = float('inf')  # Higher is worse
            program.fitness_scores["maintainability_index"] = 0.0  # Lower is worse

        # Functional Evaluation (I/O examples) - run if examples exist, regardless of mode (for regression)
            if task.input_output_examples:
                logger.debug(f"Executing program {program.id} against {len(task.input_output_examples)} test cases.")
                execution_results, execution_error = await self._execute_code_safely(program.code,
                                                                                     task_for_examples=task)

                if execution_error:
                    logger.warning(f"Execution error for program {program.id}: {execution_error}")
                    program.errors.append(f"Execution Error: {execution_error}")
                    program.fitness_scores["correctness"] = 0.0
                elif execution_results:
                    logger.debug(f"Execution results for program {program.id}: {execution_results}")
                    correctness, passed_tests, total_tests = self._assess_correctness(execution_results,
                                                                                      task.input_output_examples)
                    program.fitness_scores["correctness"] = correctness
                    program.fitness_scores["passed_tests"] = float(passed_tests)
                    program.fitness_scores["total_tests"] = float(total_tests)
                    if "average_runtime_ms" in execution_results and correctness > 0:  # Ensure positive correctness before assigning runtime
                        program.fitness_scores["runtime_ms"] = execution_results["average_runtime_ms"]
                    else:  # If correctness is 0, runtime is effectively infinite or irrelevant for fitness
                        program.fitness_scores["runtime_ms"] = float('inf')

                    logger.info(
                        f"Program {program.id} I/O correctness: {correctness * 100:.2f}% ({passed_tests}/{total_tests} tests passed)")
                    if correctness < 1.0:
                        program.errors.append(
                            f"Failed {total_tests - passed_tests} out of {total_tests} I/O test cases.")
                else:
                    logger.warning(
                        f"Execution of program {program.id} yielded no results and no specific error message.")
                    program.errors.append("Execution Error: Unknown issue, no results from sandbox.")
                    program.fitness_scores["correctness"] = 0.0
                    program.fitness_scores["runtime_ms"] = float('inf')

            elif task.improvement_mode == "task_focused":
                logger.warning(
                    f"Task {task.id} is 'task_focused' but has no input/output examples. Cannot assess correctness.")
                program.errors.append(
                    "Evaluation: Task is 'task_focused' but no I/O examples provided for correctness assessment.")

            if not program.errors:
                program.status = "evaluated"
            else:
                program.status = "failed_evaluation"

            logger.info(
                f"Evaluation complete for program {program.id}. Status: {program.status}, Fitness: {program.fitness_scores}")
            return program

        async def execute(self, program: Program, task: TaskDefinition) -> Program:  # Unchanged
            return await self.evaluate_program(program, task)