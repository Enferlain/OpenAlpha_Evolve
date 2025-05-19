from typing import Optional, Dict, Any, List
import logging

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)


class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        logger.info(
            f"PromptDesignerAgent initialized for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")

    # --- NEW HELPER METHOD (v1.0.0 for this specific helper) ---
    def _format_static_analysis_feedback_for_llm(self, program_errors: List[str]) -> str:  # Method_v1.0.0
        """
        Formats static analysis feedback (currently Ruff) from program.errors for the LLM.
        """
        # Filter for messages prefixed with "Ruff-" (or any other static analyzer we might add)
        static_analysis_messages = [err for err in program_errors if err.strip().lower().startswith("ruff-")]

        if not static_analysis_messages:
            # If we want to explicitly say no static issues were found by Ruff:
            # return "Static Analysis Feedback (from Ruff):\n  No specific issues reported by Ruff."
            # Or, return an empty string if no feedback means nothing to show.
            # Let's go with explicit for clarity in the prompt.
            return "Static Analysis Feedback (from Ruff):\n  No specific issues reported."

        # Limit the number of messages to avoid overwhelming the LLM
        max_messages_to_show = 7  # Configurable, or based on message severity if we parse that
        display_messages = static_analysis_messages[:max_messages_to_show]

        formatted_errors = ["  - " + msg for msg in display_messages]

        if len(static_analysis_messages) > max_messages_to_show:
            formatted_errors.append(
                f"  ... and {len(static_analysis_messages) - max_messages_to_show} more static analysis issues.")

        return "Static Analysis Feedback (from Ruff):\n" + "\n".join(formatted_errors)

    # Helper: _format_input_output_examples (as before)
    def _format_input_output_examples(self, task: TaskDefinition) -> str:
        if not task.input_output_examples:
            return "No input/output examples provided."
        formatted_examples = []
        for i, example in enumerate(task.input_output_examples):
            input_str = str(example.get('input'))
            output_str = str(example.get('output'))
            formatted_examples.append(f"Example {i + 1}:\n  Input: {input_str}\n  Expected Output: {output_str}")
        return "\n".join(formatted_examples)

    # Helper: _format_ancestral_summary_for_prompt (as before)
    def _format_ancestral_summary_for_prompt(self, ancestral_summary: Optional[List[Dict[str, Any]]]) -> str:
        if not ancestral_summary:
            return ""
        history_lines = [f"  - Gen {s['generation']} ({s['creation_method']}): {s['outcome_summary']}" for s in
                         ancestral_summary]
        if not history_lines: return ""
        return "Brief History of Recent Ancestors (to consider and avoid repeating less successful paths):\n" + "\n".join(
            history_lines) + "\n\n"

    def design_initial_prompt(self, task: TaskDefinition) -> str:  # Added task: TaskDefinition _v3
        logger.info(f"Designing initial prompt for task: {task.id}")
        prompt = (
            f"You are an expert Python programmer. Your task is to write a Python function based on the following specifications.\n\n"
            f"Task Description: {task.description}\n\n"  # Use task.description
            f"Function to Implement: `{task.function_name_to_evolve}`\n\n"  # Use task.function_name_to_evolve
            f"Input/Output Examples:\n"
            # We need to call _format_input_output_examples with the 'task' object
            # or modify _format_input_output_examples to take 'task'
            f"{self._format_input_output_examples(task)}\n\n"  # Pass task here _v3
            f"Evaluation Criteria: {task.evaluation_criteria}\n\n"  # Use task.evaluation_criteria
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use any other external libraries or packages.\n\n"  # Use task.allowed_imports
            f"Your Response Format:\n"
            f"Please provide *only* the complete Python code for the function `{task.function_name_to_evolve}`. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
        )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # Modified: design_crossover_prompt (no direct Ruff feedback to LLM, but parents' general quality matters)
    def design_crossover_prompt(self, task: TaskDefinition, parent_program1: Program, parent_program2: Program) -> str: # v1.0.1 (minor logging)
        logger.info(f"Designing crossover prompt for task: {task.id} using parents {parent_program1.id} and {parent_program2.id}")
        # For crossover, we might not explicitly list Ruff errors of parents, but the LLM
        # should still aim for high quality. The prompt implies combining "best features".
        # If parents had Ruff errors, their "fitness_scores" reflect that indirectly.

        # Helper to format parent info concisely (could include a very brief quality note if desired)
        def format_parent_info(parent_prog: Program, number: int) -> str:
            # Simplified: just code. If detailed feedback per parent is needed here, it can be added.
            return (
                f"--- Parent {number} (ID: {parent_prog.id}, Gen: {parent_prog.generation}) ---\n"
                f"Code:\n```python\n{parent_prog.code}\n```\n"
                # If we want to give summary of parent's quality:
                # f"Note: Parent {number} had {parent_prog.fitness_scores.get('ruff_violations',0)} Ruff issues and {parent_prog.fitness_scores.get('correctness',0)*100:.0f}% correctness.\n"
            )
        parent1_info = format_parent_info(parent_program1, 1)
        parent2_info = format_parent_info(parent_program2, 2)

        prompt = (
            f"You are an expert Python programmer specializing in synthesizing optimal solutions.\n"
            f"Your task is to perform a 'genetic crossover' on two parent Python functions to create a new, potentially superior child function.\n\n"
            f"Overall Task Description: {task.description}\n"
            f"Function to Evolve: `{task.function_name_to_evolve}`\n"
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use any other external libraries or packages.\n\n"
            f"{parent1_info}\n"
            f"{parent2_info}\n"
            f"--- Crossover Instructions ---\n"
            f"Analyze both Parent 1 and Parent 2. Your goal is to create a NEW child function that:\n"
            f"1. Solves the 'Overall Task Description' effectively.\n"
            f"2. Combines the best features, logic, or approaches from *both* parent functions.\n"
            f"3. Aims for high code quality (e.g., few static analysis issues, good readability, efficiency).\n"
            f"4. Adheres to the specified `function_name_to_evolve` and `allowed_imports`.\n\n"
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new child function. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
        )
        logger.debug(f"Designed crossover prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # --- MODIFIED: _format_evaluation_feedback (v6 for Ruff and cleaner structure) ---
    def _format_evaluation_feedback(self, task: TaskDefinition, program: Program) -> str:  # Method_v6
        """
        Formats evaluation feedback for a program, focusing on I/O tests, execution errors,
        and static analysis feedback (Ruff).
        This method primarily sources data from the program object's .fitness_scores and .errors.
        """
        feedback_parts = []

        # 1. I/O Test and Execution Feedback
        correctness = program.fitness_scores.get("correctness")
        runtime = program.fitness_scores.get("runtime_ms")
        passed_tests = program.fitness_scores.get("passed_tests")
        total_tests = program.fitness_scores.get("total_tests")

        # Execution errors (non-Ruff errors from program.errors)
        execution_errors = [err for err in program.errors if not err.strip().lower().startswith("ruff-")]

        if correctness is not None:
            feedback_parts.append(f"- Correctness (from I/O tests): {correctness * 100:.2f}%")
            if total_tests is not None and total_tests > 0:  # Check if total_tests is valid
                feedback_parts.append(
                    f"  - Passed {int(passed_tests if passed_tests is not None else 0)} out of {int(total_tests)} I/O test cases.")

        if runtime is not None and runtime != float('inf'):
            feedback_parts.append(f"- Runtime (from I/O tests): {runtime:.2f} ms")

        if execution_errors:
            error_messages_str = "\n".join([f"  - {str(e)}" for e in execution_errors if str(e).strip()])
            if error_messages_str.strip():
                feedback_parts.append(f"- Other Issues Encountered During Execution:\n{error_messages_str}")

        # 2. Static Analysis Metric Feedback (Ruff, Radon - from fitness_scores)
        # These are the numerical scores that the SelectionController also uses.
        # The detailed Ruff messages are handled by _format_static_analysis_feedback_for_llm.
        # Here, we report the *summary scores* if they are primary focus metrics or generally informative.

        ruff_violations = program.fitness_scores.get("ruff_violations")
        if ruff_violations is not None and ruff_violations != float('inf'):
            feedback_parts.append(f"- Ruff Static Analysis Violations: {int(ruff_violations)} (lower is better)")

        cyclomatic_complexity_avg = program.fitness_scores.get("cyclomatic_complexity_avg")
        if cyclomatic_complexity_avg is not None and cyclomatic_complexity_avg != float('inf'):
            feedback_parts.append(
                f"- Average Cyclomatic Complexity: {cyclomatic_complexity_avg:.2f} (lower is generally better)")

        maintainability_index = program.fitness_scores.get("maintainability_index")
        # Assuming -1.0 was default for "not run" or error for MI
        if maintainability_index is not None and maintainability_index > 0:  # MI usually 0-100
            feedback_parts.append(
                f"- Maintainability Index: {maintainability_index:.2f} (higher is generally better, 0-100)")

        # 3. Conditional notes based on overall status (if no direct execution errors were primary)
        # This part might be redundant if the detailed errors already cover the situation.
        # Consider if this is still needed or if the specific error lists are sufficient.
        # For now, keeping a simplified version.
        if not execution_errors:  # Only add these general notes if no direct execution errors were the main issue
            if correctness is not None:
                if correctness < 1.0:
                    feedback_parts.append(
                        "- Note: The code did not achieve 100% correctness on I/O tests. Please review logic.")
                elif correctness == 1.0:  # Correctness is 1.0
                    if task.improvement_mode == "task_focused":
                        feedback_parts.append(
                            "- Note: Code achieved 100% correctness! Focus on other quality aspects (e.g., reducing Ruff violations, efficiency) or alternative logic.")
                    elif task.improvement_mode == "general_refinement":
                        feedback_parts.append(
                            "- Note: Code maintained 100% correctness (if I/O tests provided). Focus on improving other quality metrics (Ruff, Complexity, MI) and specific directives.")

        if not feedback_parts:
            # This case should be rare if fitness_scores are always populated
            return "No specific evaluation feedback details were captured for the previous version. Please attempt a general improvement based on the task goals."

        return "Summary of the previous version's evaluation:\n" + "\n".join(feedback_parts)

    # --- MODIFIED: design_mutation_prompt (v3 for Ruff feedback) ---
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program,
                               evaluation_feedback: Optional[Dict[str, Any]] = None,
                               # This 'evaluation_feedback' might be less critical now
                               ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:  # Method_v3
        logger.info(
            f"Designing mutation prompt for program: {parent_program.id} (Gen: {parent_program.generation}, Mode: {task.improvement_mode})")

        # Primary feedback for LLM now comes from parent_program.errors (for Ruff) and I/O test results.
        # The 'evaluation_feedback' dict passed in might be redundant if all info is on Program object.
        # For now, let's assume parent_program has updated .errors and .fitness_scores.

        # 1. Feedback from I/O Tests (Correctness & Runtime)
        io_feedback_parts = []
        correctness = parent_program.fitness_scores.get("correctness")
        if correctness is not None:
            io_feedback_parts.append(f"- Correctness (from I/O tests): {correctness * 100:.2f}%")
            passed = parent_program.fitness_scores.get("passed_tests", 0)
            total = parent_program.fitness_scores.get("total_tests", 0)
            if total > 0:
                io_feedback_parts.append(f"  - Passed {int(passed)} out of {int(total)} I/O test cases.")

        runtime = parent_program.fitness_scores.get("runtime_ms")
        if runtime is not None and runtime != float('inf'):
            io_feedback_parts.append(f"- Runtime (from I/O tests): {runtime:.2f} ms")

        # Add execution errors if they are not Ruff errors
        execution_errors = [err for err in parent_program.errors if not err.strip().lower().startswith("ruff-")]
        if execution_errors:
            io_feedback_parts.append(
                f"- Other Issues Encountered During Execution:\n" + "\n".join([f"  - {e}" for e in execution_errors]))

        io_feedback_summary = "I/O Test and Execution Feedback:\n" + ("\n".join(
            io_feedback_parts) if io_feedback_parts else "  No specific I/O test issues reported, or no tests run.")

        # 2. Static Analysis Feedback (from Ruff, using the new helper)
        static_analysis_summary = self._format_static_analysis_feedback_for_llm(parent_program.errors)

        # 3. Ancestral History
        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)

        diff_instructions = (
            "Your Response Format:\n"
            "Propose improvements to the 'Current Code' below by providing your changes as a sequence of diff blocks. "  # ... (as before)
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Current Code'.\n"
            "- If adding new code, SEARCH can be an adjacent line or a comment indicating location.\n"
            "- If deleting code, REPLACE should be empty.\n"
            "- Provide all changes as diff blocks. No other text or explanations."
        )
        prompt_header = f"You are an expert Python programmer. Your task is to improve an existing Python function based on its previous performance, historical attempts, and the overall goal.\n\n"
        current_code_section = f"Current Code (Version from Generation {parent_program.generation}):\n```python\n{parent_program.code}\n```\n\n"
        allowed_imports_section = f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use other external libraries or packages.\n\n"

        prompt_parts = [
            prompt_header,
            f"Overall Task Context: {task.description}\nFunction of Interest: `{task.function_name_to_evolve}`\n",
            allowed_imports_section,
            current_code_section,
            io_feedback_summary,  # IO feedback first
            "\n" + static_analysis_summary,  # Then Ruff feedback
            "\n" + historical_context_section,  # Then history
        ]

        if task.improvement_mode == "general_refinement":
            improvement_goal_intro = "Your Improvement Goal (General Refinement):\n"
            directives_section = f"Specific Directives: {task.specific_improvement_directives}\n" if task.specific_improvement_directives else ""
            metrics_str = ", ".join(
                task.primary_focus_metrics) if task.primary_focus_metrics else "general code quality"
            metrics_section = f"Primary Focus Metrics: Aim to improve {metrics_str}.\n"
            improvement_goal_details = (
                f"Based on all feedback, your goal is to refine this code. "
                f"{directives_section}"
                f"{metrics_section}"
                f"Ensure changes do not break existing I/O functionality.\n\n"
            )
            prompt_parts.extend([improvement_goal_intro, improvement_goal_details])
        else:  # "task_focused"
            improvement_goal_intro = "Your Improvement Goal (Task Focused):\n"
            improvement_goal_details = (
                f"Based on all feedback, your goal is to improve function `{task.function_name_to_evolve}` "
                f"to better solve: {task.description}\n"
                f"Prioritize fixing any I/O correctness or execution errors, then static analysis issues. "
                f"If correct and clean, focus on efficiency or robustness. "
                f"Original criteria: {task.evaluation_criteria}\n\n"
            )
            prompt_parts.extend([improvement_goal_intro, improvement_goal_details])

        prompt_parts.append(diff_instructions)
        prompt = "".join(prompt_parts)

        logger.debug(
            f"Designed mutation prompt (mode: {task.improvement_mode}, requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # --- MODIFIED: design_bug_fix_prompt (v3 for Ruff feedback) ---
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_message: str,
                              # error_message is the primary execution error
                              execution_output: Optional[str] = None,
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:  # Method_v3
        logger.info(
            f"Designing bug-fix prompt for program: {program.id} (Gen: {program.generation}, Mode: {task.improvement_mode})")

        # Static analysis feedback (Ruff issues from program.errors)
        # Filter out the main `error_message` if it's already in program.errors to avoid duplication
        other_static_errors = [e for e in program.errors if
                               e != error_message and e.strip().lower().startswith("ruff-")]
        static_analysis_summary = self._format_static_analysis_feedback_for_llm(other_static_errors)
        if not other_static_errors:  # If the only errors were execution errors, explicitly say no Ruff issues
            static_analysis_summary = "Static Analysis Feedback (from Ruff):\n  No specific Ruff issues reported beyond the main execution error."

        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        output_segment = f"Execution Output (stdout/stderr that might be relevant):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the error message itself.\n"
        diff_instructions = (
            "Your Response Format:\n"  # ... (as before) ...
            "Propose fixes to the 'Buggy Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Buggy Code'.\n"
            "- Provide all suggested changes as one or more such diff blocks. No other text or explanations."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to fix a bug in an existing Python function, considering previous attempts and static analysis feedback.\n\n"
            f"Overall Task Description: {task.description}\n"
            f"Function to Fix: `{task.function_name_to_evolve}`\n"
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Buggy Code (Version from Generation {program.generation}):\n```python\n{program.code}\n```\n\n"
            f"Primary Error Encountered During Execution: {error_message}\n"
            f"{output_segment}\n"
            f"{static_analysis_summary}\n\n"  # Add Ruff feedback here
            f"{historical_context_section}"
            f"Your Goal:\n"
            f"Analyze the 'Buggy Code', the 'Primary Error', 'Execution Output', any 'Static Analysis Feedback', and 'Brief History...' to identify and fix the bug(s). "
            f"The corrected function must adhere to the overall task description and allowed imports.\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # --- MODIFIED: design_failed_diff_fallback_prompt (v3 for Ruff context) ---
    def design_failed_diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,
                                           # This summary should ideally include why diff failed + original goal
                                           ancestral_summary: Optional[
                                               List[Dict[str, Any]]] = None) -> str:  # Method_v3
        logger.info(
            f"Designing failed-diff fallback prompt for program: {original_program.id} (Gen: {original_program.generation})")

        # Include Ruff feedback for the original_program if it's relevant to the fallback
        static_analysis_summary = self._format_static_analysis_feedback_for_llm(original_program.errors)
        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)

        prompt = (
            f"You are an expert Python programmer. Your previous attempt to provide improvements for the following code via a diff format was unsuccessful or resulted in no changes.\n\n"
            f"Overall Task Description: {task.description}\n"
            f"Function of Interest: `{task.function_name_to_evolve}`\n"
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Original Code (that was attempted to be diffed):\n```python\n{original_program.code}\n```\n\n"
            f"{static_analysis_summary}\n\n"  # Add Ruff feedback on the original code
            f"Summary of Previous Attempt's Goal: {previous_attempt_summary}\n\n"
            f"{historical_context_section}"
            f"Your New Goal:\n"
            f"Please provide the *complete, fully corrected/improved version* of the function `{task.function_name_to_evolve}`. "
            f"Ensure your new version addresses the objectives from the 'Previous Attempt's Goal', any 'Static Analysis Feedback', and learns from 'Brief History...'.\n\n"
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new version of the function. "
            f"No surrounding text, explanations, or markdown code fences."
        )
        logger.debug(f"Designed failed-diff fallback prompt:\n{prompt}")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "PromptDesignerAgent.execute() is not the primary way to use this agent. Call specific design methods.")