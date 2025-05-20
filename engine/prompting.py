# prompt_designer/prompting.py
# Version: 1.1.0 (Added LLM Reviewe Prompt Design)

from typing import Optional, Dict, Any, List
import logging

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)


class PromptStudio(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition): # TaskDefinition might not be needed for all prompts if passed directly
        super().__init__()
        # Storing task_definition here might be useful for general context,
        # but specific methods will receive the relevant task/program objects.
        self.task_definition_context = task_definition # Store for general reference if needed
        logger.info(
            f"PromptStudio initialized. Context Task ID: {self.task_definition_context.id if self.task_definition_context else 'None'}"
        )

    # --- NEW HELPER METHOD (v1.0.0 for this specific helper) ---
    def _format_analysis_feedback(self, program_errors: List[str]) -> str:  # Method_v1.0.0
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

    # Helper: _format_io_examples (as before)
    def _format_io_examples(self, task: TaskDefinition) -> str:
        if not task.io_examples:
            return "No input/output examples provided."
        formatted_examples = []
        for i, example in enumerate(task.io_examples):
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

    def initial_prompt(self, task: TaskDefinition) -> str:  # Method_v_blueprint_1.0.0
        logger.info(f"Designing initial prompt for task: {task.id} (Blueprint Aligned)")

        prompt_parts = [
            f"You are an expert and creative Python programmer. Your goal is to develop a solution based on the following high-level description and requirements.\n\n"
            f"## Overall Goal & Context:\n{task.description}\n\n"
        ]

        if task.target_solution:
            prompt_parts.append(f"## Expected Solution Output:\n{task.target_solution}\n\n")
        elif task.evolve_function:  # Fallback or hint if no target_solution
            prompt_parts.append(
                f"## Primary Function to Implement (if applicable):\n`{task.evolve_function}`\n\n")

        if task.initial_seed:
            prompt_parts.append(
                f"## Starting Point (Seed Code or Ideas to Build Upon/Consider):\n"
                f"```text\n{task.initial_seed}\n```\n\n"  # Use text block for ideas too
            )

        if task.suggested_imports:
            prompt_parts.append(
                f"Suggested Python libraries you might find useful (not strict constraints unless implied by the task): {', '.join(task.suggested_imports)}.\n\n")
        elif task.allowed_imports:  # If only old allowed_imports is used, treat as suggestions
            prompt_parts.append(
                f"Consider using these standard libraries if helpful: {', '.join(task.allowed_imports)}.\n\n")

        # Mention how it will be evaluated
        if task.ai_review_criteria:
            prompt_parts.append(
                f"## Evaluation Focus:\nYour solution will be primarily evaluated by an Ai Reviewer based on the following guidelines. Strive to meet these criteria:\n"
                f"{task.ai_review_criteria}\n\n"
            )
        elif task.io_examples:  # Fallback to I/O examples if no ai review guidelines
            formatted_examples = self._format_io_examples(task)
            prompt_parts.append(
                f"## Evaluation Focus:\nYour solution will be evaluated for correctness based on input/output examples like these:\n{formatted_examples}\n\n"
            )

        prompt_parts.append(
            f"Your Response Format:\n"
            f"Please provide *only* the complete Python code for the solution. "
            f"The code should be self-contained or rely on the suggested imports if appropriate. "
            f"Do not include any surrounding text, explanations, comments outside the code structure, or markdown code fences (like ```python or ```)."
        )

        prompt = "".join(prompt_parts)
        logger.debug(f"Designed blueprint-aligned initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _format_execution_summary(self,
                                                   execution_summary: Dict[str, Any]) -> str:  # NEW HELPER (as before)
        parts = []
        runs_ok = execution_summary.get("runs_without_error")
        if runs_ok is not None:
            parts.append(
                f"- Standalone Execution Attempt: {'Script ran successfully.' if runs_ok else 'Script failed to run or crashed.'}")

        runtime_ms = execution_summary.get("standalone_script_runtime_ms")
        if runtime_ms is not None and runtime_ms != float('inf'):
            parts.append(f"  - Runtime (if successful): {runtime_ms:.2f} ms")

        exec_stdout = execution_summary.get("stdout", "").strip()
        exec_stderr = execution_summary.get("stderr", "").strip()

        log_parts = []
        if exec_stdout:
            log_parts.append(f"STDOUT:\n{exec_stdout[:1000]}{'...' if len(exec_stdout) > 1000 else ''}")
        if exec_stderr:
            log_parts.append(f"STDERR:\n{exec_stderr[:1000]}{'...' if len(exec_stderr) > 1000 else ''}")

        if log_parts:
            parts.append("- Key Execution Logs/Errors:\n  " + "\n  ".join(log_parts))
        elif runs_ok is False:
            parts.append(
                "- Key Execution Logs/Errors: Execution failed, specific logs not summarized here (check program.errors).")

        ruff_violations = execution_summary.get("ruff_violations")
        if ruff_violations is not None and ruff_violations != float('inf'):
            parts.append(f"- Static Analysis (Ruff): Found {int(ruff_violations)} issues.")

        correctness = execution_summary.get("correctness")
        if correctness is not None:
            parts.append(f"- I/O Test Correctness (if applicable): {correctness * 100:.1f}%")

        if not parts:
            return "No specific automated check observations were provided."
        return "\n".join(parts)

    def crossover_prompt(self, task: TaskDefinition, parent_program1: Program,
                                parent_program2: Program) -> str:  # Method_v_blueprint_1.0.0
        logger.info(
            f"Designing blueprint-aligned crossover prompt for task: {task.id} using parents {parent_program1.id} and {parent_program2.id}")

        def format_parent_info_for_crossover(parent_prog: Program, number: int) -> str:
            parts = [
                f"--- Parent {number} (ID: {parent_prog.id}, Gen: {parent_prog.generation}) ---",
                f"Code:\n```python\n{parent_prog.code}\n```"
            ]
            if parent_prog.ai_review_feedback:
                parts.append(
                    f"Parent {number} Ai Reviewer Assessment (Score: {parent_prog.fitness_scores.get('ai_review_score', 'N/A')}/10):\n{parent_prog.ai_review_feedback}"
                )
            # Add brief summary of other key metrics if desired
            # ruff_v = parent_prog.fitness_scores.get('ruff_violations', 'N/A')
            # correctness_v = parent_prog.fitness_scores.get('correctness', -1.0) * 100
            # parts.append(f"Parent {number} Stats: Ruff Issues: {ruff_v}, I/O Correctness: {correctness_v if correctness_v >=0 else 'N/A'}%.")
            return "\n".join(parts) + "\n"

        parent1_info = format_parent_info_for_crossover(parent_program1, 1)
        parent2_info = format_parent_info_for_crossover(parent_program2, 2)

        prompt_parts = [
            f"You are an expert Python programmer specializing in synthesizing optimal solutions by combining ideas.\n"
            f"Your task is to perform a 'genetic crossover' on two parent Python solutions to create a new, potentially superior child solution.\n\n"
            f"## Overall Goal & Context:\n{task.description}\n"
        ]
        if task.target_solution:
            prompt_parts.append(f"## Expected Solution Output:\n{task.target_solution}\n")
        if task.ai_review_criteria:
            prompt_parts.append(
                f"## Primary Evaluation Guidelines for Child (Aim to excel here):\n{task.ai_review_criteria}\n"
            )
        prompt_parts.append("\n")

        prompt_parts.append(parent1_info)
        prompt_parts.append(parent2_info)

        prompt_parts.append(
            f"--- Crossover Instructions ---\n"
            f"Analyze both Parent 1 and Parent 2, including their code and any Ai Reviewer assessments.\n"
            f"Your goal is to create a NEW child solution that:\n"
            f"1. Effectively solves the 'Overall Goal & Context'.\n"
            f"2. Synthesizes the best features, logic, or approaches from *both* parents, while AVOIDING flaws identified in either parent by the Ai Reviewer.\n"
            f"3. Aims to satisfy the 'Primary Evaluation Guidelines' better than both parents.\n"
            f"4. Adheres to any specified solution structure (e.g., target_solution) and suggested imports.\n\n"
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new child solution. "
            f"No surrounding text, explanations, comments outside the code, or markdown code fences."
        )

        prompt = "".join(prompt_parts)
        logger.debug(f"Designed blueprint-aligned crossover prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
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
        # The detailed Ruff messages are handled by _format_analysis_feedback.
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

    def mutation_prompt(self, task: TaskDefinition, parent_program: Program,
                               ancestral_summary: Optional[List[Dict[str, Any]]] = None
                               ) -> str:  # Method_v_blueprint_1.0.1 (Signature updated)
        logger.info(
            f"Designing blueprint-aligned mutation prompt for program: {parent_program.id} (Gen: {parent_program.generation})"
        )

        prompt_parts = [
            f"You are an expert Python programmer. Your task is to REFINE and IMPROVE an existing Python solution based on previous evaluations and the overall task goals.\n\n"
            f"## Overall Goal & Context:\n{task.description}\n\n"
        ]
        if task.target_solution:
            prompt_parts.append(f"## Expected Solution Output:\n{task.target_solution}\n\n")

        prompt_parts.append(
            f"## Current Code (Version from Generation {parent_program.generation}):\n```python\n{parent_program.code}\n```\n\n")

        prompt_parts.append(f"## Feedback on Current Code:\n")

        if parent_program.ai_review_feedback:
            prompt_parts.append(
                f"### Ai Reviewer's Assessment (Overall Score: {parent_program.fitness_scores.get('ai_review_score', 'N/A')}/10):\n"
                f"{parent_program.ai_review_feedback}\n\n"
            )
        elif task.ai_review_criteria:
            prompt_parts.append(
                f"### Reminder of Evaluation Guidelines (Aim for these):\n"
                f"{task.ai_review_criteria}\n\n"
            )

        exec_status = "Ran successfully" if parent_program.fitness_scores.get(
            "runs_without_error") else "Failed or crashed during basic execution"
        prompt_parts.append(f"- Basic Execution Status: {exec_status}\n")
        if parent_program.fitness_scores.get("correctness") is not None:
            prompt_parts.append(
                f"- I/O Test Correctness: {parent_program.fitness_scores.get('correctness', 0.0) * 100:.1f}%\n")

        static_analysis_summary = self._format_analysis_feedback(parent_program.errors)
        prompt_parts.append(f"{static_analysis_summary}\n")

        critical_errors = [e for e in parent_program.errors if not e.startswith(
            "Ruff-") and "Execution Log (STDOUT" not in e and "Execution Log (STDERR" not in e]
        if critical_errors:
            prompt_parts.append(f"### Other Critical Issues from Last Attempt:\n" + "\n".join(
                [f"  - {ce}" for ce in critical_errors]) + "\n\n")

        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        if historical_context_section:
            prompt_parts.append(historical_context_section)

        prompt_parts.append(
            f"## Your Improvement Goal:\n"
            f"Based on all the feedback above (especially the Ai Reviewer's assessment if available), your goal is to significantly improve the 'Current Code'.\n"
            f"Address the weaknesses pointed out by the Ai Reviewer and fix any reported errors or static analysis issues.\n"
            f"Strive to better meet the 'User's Evaluation Guidelines' (mentioned in the Ai Reviewer's feedback or the task context).\n"
        )
        if task.refine_goals:
            prompt_parts.append(f"Also consider these specific directives: {task.refine_goals}\n")
        if task.primary_focus_metrics:
            prompt_parts.append(
                f"Consider improving these metrics if possible: {', '.join(task.primary_focus_metrics)}.\n")

        prompt_parts.append(
            f"\nYour Response Format:\n"  # Diff instructions as before
            f"Propose improvements by providing changes as a sequence of diff blocks. "
            f"Each diff block must follow this exact format:\n"
            f"<<<<<<< SEARCH\n"
            f"# Exact original code lines to be found and replaced\n"
            f"=======\n"
            f"# New code lines to replace the original\n"
            f">>>>>>> REPLACE\n\n"
            f"- The SEARCH block must be an *exact* segment from the 'Current Code'.\n"
            f"- Provide all changes as diff blocks. No other text or explanations.\n"
        )

        prompt = "".join(prompt_parts)
        logger.debug(
            f"Designed blueprint-aligned mutation prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # In PromptStudio
    def bugfix_prompt(self, task: TaskDefinition, program: Program,  # Removed error_message, execution_output
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None
                              ) -> str:  # Method_v_blueprint_1.0.1 (Signature updated, typo fixed)
        logger.info(
            f"Designing blueprint-aligned bug-fix prompt for program: {program.id} (Gen: {program.generation})"
        )

        prompt_parts = [
            f"You are an expert Python programmer. Your task is to FIX CRITICAL ISSUES in an existing Python solution based on previous evaluations and the overall task goals.\n\n"
            f"## Overall Goal & Context:\n{task.description}\n\n"
        ]
        if task.target_solution:
            prompt_parts.append(f"## Expected Solution Output:\n{task.target_solution}\n\n")

        prompt_parts.append(
            f"## Buggy Code (Version from Generation {program.generation}):\n```python\n{program.code}\n```\n\n")

        prompt_parts.append(f"## Feedback on Buggy Code:\n")

        primary_failure_reason = f"The code failed with status: {program.status}."
        critical_errors_from_program = [e for e in program.errors if not e.startswith(
            "Ruff-") and "Execution Log (STDOUT" not in e and "Execution Log (STDERR" not in e]

        if "failed_evaluation_execution" in program.status and critical_errors_from_program:
            primary_failure_reason += f"\n  - Key Error: {critical_errors_from_program[0]}"
        elif "failed_evaluation_syntax" in program.status and critical_errors_from_program:
            primary_failure_reason += f"\n  - Syntax Error: {critical_errors_from_program[0]}"
        elif program.ai_review_feedback and (program.fitness_scores.get('ai_review_score', 10) <= 3):
            primary_failure_reason = (
                f"The Ai Reviewer gave a very low score ({program.fitness_scores.get('ai_review_score')}/10) "
                f"indicating critical flaws. Ai Reviewer's Justification:\n{program.ai_review_feedback}")

        prompt_parts.append(f"### Primary Problem:\n{primary_failure_reason}\n\n")

        # *** FIXED TYPO HERE: program.fitness_scores instead of parent_program.fitness_scores ***
        if program.ai_review_feedback and not (program.fitness_scores.get('ai_review_score', 10) <= 3):
            prompt_parts.append(
                f"### Ai Reviewer's Assessment (Overall Score: {program.fitness_scores.get('ai_review_score', 'N/A')}/10):\n"  # Was parent_program
                f"{program.ai_review_feedback}\n\n"
            )

        static_analysis_summary = self._format_analysis_feedback(program.errors)
        prompt_parts.append(f"{static_analysis_summary}\n")

        if critical_errors_from_program and (not ("failed_evaluation_execution" in program.status) and not (
                "failed_evaluation_syntax" in program.status)):
            prompt_parts.append(f"### Other Critical Issues from Last Attempt:\n" + "\n".join(
                [f"  - {ce}" for ce in critical_errors_from_program]) + "\n\n")

        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        if historical_context_section:
            prompt_parts.append(historical_context_section)

        prompt_parts.append(
            f"## Your Bug-Fix Goal:\n"
            f"Analyze the 'Buggy Code' and all the feedback. Your primary goal is to FIX the identified 'Primary Problem'.\n"
            f"Also address any other reported errors or static analysis issues.\n"
            f"The corrected solution must work as intended by the 'Overall Goal & Context'.\n"
        )

        prompt_parts.append(
            f"\nYour Response Format:\n"  # Diff instructions as before
            f"Propose fixes by providing changes as a sequence of diff blocks. "
            f"Each diff block must follow this exact format:\n"
            f"<<<<<<< SEARCH\n"
            f"# Exact original code lines to be found and replaced\n"
            f"=======\n"
            f"# New code lines to replace the original\n"
            f">>>>>>> REPLACE\n\n"
            f"- The SEARCH block must be an *exact* segment from the 'Buggy Code'.\n"
            f"- Provide all suggested changes as one or more such diff blocks. No other text or explanations.\n"
        )

        prompt = "".join(prompt_parts)
        logger.debug(
            f"Designed blueprint-aligned bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,  # Goal of the failed diff
                                           ancestral_summary: Optional[
                                               List[Dict[str, Any]]] = None) -> str:  # Method_v_blueprint_1.0.0
        logger.info(
            f"Designing blueprint-aligned failed-diff fallback prompt for program: {original_program.id} (Gen: {original_program.generation})"
        )

        prompt_parts = [
            f"You are an expert Python programmer. Your previous attempt to provide improvements for the following code via a diff format was unsuccessful or resulted in no changes. You will now provide the complete, improved code.\n\n"
            f"## Overall Goal & Context:\n{task.description}\n"
        ]
        if task.target_solution:
            prompt_parts.append(f"## Expected Solution Output:\n{task.target_solution}\n")

        prompt_parts.append(
            f"\n## Original Code (that was attempted to be diffed):\n```python\n{original_program.code}\n```\n\n")

        # Feedback on Original Code
        prompt_parts.append(f"## Feedback on Original Code (that the failed diff was targeting):\n")
        if original_program.ai_review_feedback:
            prompt_parts.append(
                f"### Ai Reviewer's Assessment (Score: {original_program.fitness_scores.get('ai_review_score', 'N/A')}/10):\n{original_program.ai_review_feedback}\n\n"
            )

        static_analysis_summary = self._format_analysis_feedback(original_program.errors)
        prompt_parts.append(f"{static_analysis_summary}\n")  # Ruff issues

        critical_errors = [e for e in original_program.errors if not e.startswith(
            "Ruff-") and "Execution Log (STDOUT" not in e and "Execution Log (STDERR" not in e]
        if critical_errors:
            prompt_parts.append(f"### Other Critical Issues in Original Code:\n" + "\n".join(
                [f"  - {ce}" for ce in critical_errors]) + "\n\n")

        prompt_parts.append(f"## Summary of Previous (Failed Diff) Attempt's Goal:\n{previous_attempt_summary}\n\n")

        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        if historical_context_section:
            prompt_parts.append(historical_context_section)

        prompt_parts.append(
            f"## Your New Goal (Provide Full Code):\n"
            f"Please provide the *complete, fully corrected/improved version* of the solution. "
            f"Your new version must address the objectives from the 'Previous Attempt's Goal' (e.g., critiques from the Ai Reviewer, specific errors, or refinement targets) and incorporate feedback on the 'Original Code'.\n"
            f"Aim to satisfy the task's 'Evaluation Guidelines' (if provided to the ai reviewer previously) or general quality standards.\n\n"
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new version. "
            f"No surrounding text, explanations, comments outside the code, or markdown code fences."
        )

        prompt = "".join(prompt_parts)
        logger.debug(
            f"Designed blueprint-aligned failed-diff fallback prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    # --- MODIFIED: ai_review_prompt (Corrected f-string formatting) ---
    def ai_review_prompt(self,
                                task: TaskDefinition,
                                program_to_review: Program,
                                execution_summary: Dict[str, Any]
                                ) -> str:  # Method_v1.0.1
        logger.info(f"Designing ai review prompt for Program ID: {program_to_review.id}, Task ID: {task.id}")

        formatted_observations = self._format_execution_summary(execution_summary)

        # Using the (f"..." f"...") style for consistency and readability
        prompt = (
            f"You are an impartial and expert AI Code Reviewer. Your task is to meticulously evaluate a provided Python code solution based on the given task context, user guidelines, and observed behaviors.\n\n"
            f"## Task Context & Goal:\n"
            f"Problem Description:\n{task.description}\n\n"
            f"Target Solution Description:\n{task.target_solution if task.target_solution else 'Not explicitly specified, infer from problem description.'}\n\n"
            f"## User's Evaluation Guidelines:\n"
            f"Please review the code primarily based on these guidelines:\n{task.ai_review_criteria if task.ai_review_criteria else 'No specific user guidelines provided. Use general principles of good code quality, correctness for the task, and creativity.'}\n\n"
            f"## Code for Review:\n```python\n{program_to_review.code}\n```\n\n"
            f"## Observations from Automated Checks:\n{formatted_observations}\n\n"
            f"## Your Evaluation Task:\n"
            f"1. Carefully review the \"Code for Review\" in light of the \"Task Context & Goal\" and, most importantly, the \"User's Evaluation Guidelines.\"\n"
            f"2. Consider the \"Observations from Automated Checks\" as additional context. For instance, an `ImportError` might explain why a script didn't run, but the underlying logic could still be sound or flawed based on the guidelines. A crash, however, is a more severe issue.\n"
            f"3. Provide your assessment as a JSON object with the following two keys:\n"
            f"   - \"overall_score\": An integer score from 1 (Very Poor) to 10 (Excellent).\n"
            f"   - \"justification\": A concise (2-4 sentences) textual explanation for your score, highlighting key strengths and weaknesses of the code in relation to the guidelines and task.\n\n"
            f"Please output *only* the JSON object.\n"
        )
        logger.debug(
            f"Designed ai review Prompt for {program_to_review.id}:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "PromptStudio.execute() is not the primary way to use this agent. Call specific design methods.")