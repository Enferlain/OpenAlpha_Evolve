from typing import Optional, Dict, Any, List
import logging

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)


class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):  # Unchanged
        super().__init__()
        self.task_definition = task_definition
        logger.info(
            f"PromptDesignerAgent initialized for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")  # Added mode to log_v2

    def design_initial_prompt(self, task: TaskDefinition) -> str:  # Added task: TaskDefinition _v3        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
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

    def design_crossover_prompt(self, task: TaskDefinition, parent_program1: Program, parent_program2: Program) -> str: # v1.0.0
        logger.info(f"Designing crossover prompt for task: {task.id} using parents {parent_program1.id} and {parent_program2.id}")

        # Helper to format parent info concisely
        def format_parent_info(parent_prog: Program, number: int) -> str:
            feedback_summary = self._format_evaluation_feedback(task, parent_prog, parent_prog.fitness_scores) # Use its own scores
            return (
                f"--- Parent {number} (ID: {parent_prog.id}, Gen: {parent_prog.generation}) ---\n"
                f"Code:\n```python\n{parent_prog.code}\n```\n"
                f"Evaluation Feedback for this Parent:\n{feedback_summary}\n"
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
            f"2. Combines the best features, logic, or approaches from *both* parent functions. For example, if one parent is more efficient in one part and the other is more robust in another, try to get the best of both.\n"
            f"3. Learns from any weaknesses or errors mentioned in their respective 'Evaluation Feedback'.\n"
            f"4. Adheres to the specified `function_name_to_evolve` and `allowed_imports`.\n\n"
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new child function. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
        )
        logger.debug(f"Designed crossover prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _format_input_output_examples(self, task: TaskDefinition) -> str:  # Modified to accept task _v3
        if not task.input_output_examples:  # Use task.input_output_examples
            return "No input/output examples provided."
        formatted_examples = []
        for i, example in enumerate(task.input_output_examples):  # Use task.input_output_examples
            input_str = str(example.get('input'))
            output_str = str(example.get('output'))
            formatted_examples.append(f"Example {i + 1}:\n  Input: {input_str}\n  Expected Output: {output_str}")
        return "\n".join(formatted_examples)

    # MODIFIED _format_evaluation_feedback _v5 (for this file, incorporating new metrics)
    def _format_evaluation_feedback(self, task: TaskDefinition, program: Program,
                                    evaluation_feedback: Optional[Dict[str, Any]]) -> str:
        if not evaluation_feedback:  # If no feedback dict at all
            if task.improvement_mode == "general_refinement":
                return "No evaluation feedback is available for the previous version. Focus on general code quality and any specific directives provided in the main goal."
            else:  # task_focused
                return "No evaluation feedback is available for the previous version. Attempt a general improvement based on the task description and I/O examples."

        # Get scores from the evaluation_feedback dictionary (which mirrors Program.fitness_scores)
        correctness = evaluation_feedback.get("correctness", None)
        runtime = evaluation_feedback.get("runtime_ms", None)
        errors = evaluation_feedback.get("errors", [])  # Should be a list
        # stderr = evaluation_feedback.get("stderr", None) # We decided stderr is usually part of 'errors' from evaluator

        # --- New Metrics from EvaluatorAgent (v5) ---
        pylint_score = evaluation_feedback.get("pylint_score", None)
        cyclomatic_complexity_avg = evaluation_feedback.get("cyclomatic_complexity_avg", None)
        maintainability_index = evaluation_feedback.get("maintainability_index", None)
        # ---

        feedback_parts = []

        # I/O Test based feedback (if applicable)
        if correctness is not None:
            feedback_parts.append(f"- Correctness Score (from I/O tests): {correctness * 100:.2f}%")
            # Add details about passed/total tests if available
            passed_tests = evaluation_feedback.get("passed_tests", None)
            total_tests = evaluation_feedback.get("total_tests", None)
            if passed_tests is not None and total_tests is not None:
                feedback_parts.append(f"  - Passed {int(passed_tests)} out of {int(total_tests)} I/O test cases.")

        if runtime is not None and runtime != float('inf'):  # Only show runtime if it's valid
            feedback_parts.append(f"- Runtime (from I/O tests): {runtime:.2f} ms")

        # Static Analysis Metric Feedback
        if pylint_score is not None and pylint_score != -1.0:  # -1.0 was our "not run/error" default
            feedback_parts.append(f"- Pylint Score: {pylint_score:.2f}/10")

        if cyclomatic_complexity_avg is not None and cyclomatic_complexity_avg != float('inf'):  # inf was error/not run
            feedback_parts.append(
                f"- Average Cyclomatic Complexity: {cyclomatic_complexity_avg:.2f} (lower is generally better)")

        if maintainability_index is not None and maintainability_index != -1.0:  # -1.0 was error/not run
            feedback_parts.append(
                f"- Maintainability Index: {maintainability_index:.2f} (higher is generally better, range typically 0-100)")

        # Error feedback
        if errors:  # This 'errors' list comes from program.errors, populated by Evaluator
            error_messages_str = "\n".join(
                [f"  - {str(e)}" for e in errors if str(e).strip()])  # Ensure errors are strings
            if error_messages_str.strip():  # Only add if there are actual error messages
                feedback_parts.append(f"- Issues Encountered During Evaluation:\n{error_messages_str}")

        # Conditional messages based on correctness (if no explicit errors were primary)
        if not errors:  # Only add these if no direct errors were logged as primary feedback
            if correctness is not None:
                if correctness < 1.0:
                    feedback_parts.append(
                        "- Note: The code did not achieve 100% correctness on I/O tests. Please review logic for test case failures.")
                elif correctness == 1.0 and task.improvement_mode == "task_focused":
                    feedback_parts.append(
                        "- Note: The code achieved 100% correctness on I/O tests! Consider optimizing for efficiency or exploring alternative robust logic for the task.")
                elif correctness == 1.0 and task.improvement_mode == "general_refinement":
                    feedback_parts.append(
                        "- Note: The code maintained 100% correctness on I/O tests (if provided). Focus on improving other quality metrics (Pylint, Complexity, MI) and specific directives.")

        if not feedback_parts:  # If absolutely nothing to report
            if task.improvement_mode == "general_refinement":
                return "The previous version was evaluated, but no specific feedback details (I/O, static analysis, or errors) were captured. Please attempt a general improvement based on the directives and primary focus metrics."
            else:  # task_focused
                return "The previous version was evaluated, but no specific feedback details (I/O or errors) were captured. Please attempt an improvement based on the task description."

        return "Summary of the previous version's evaluation:\n" + "\n".join(feedback_parts)

    def _format_ancestral_summary_for_prompt(self, ancestral_summary: Optional[
        List[Dict[str, Any]]]) -> str:  # Method_v1.0.0 (New helper)
        if not ancestral_summary:
            return ""

        history_lines = []
        for s in ancestral_summary:
            # Example: "  - Gen 2 (mutation): Correctness: 80%, Pylint: 7.5/10"
            history_lines.append(f"  - Gen {s['generation']} ({s['creation_method']}): {s['outcome_summary']}")

        if not history_lines:
            return ""

        return (
            "Brief History of Recent Ancestors (to consider and avoid repeating less successful paths):\n"
            f"{chr(10).join(history_lines)}\n\n"  # chr(10) is newline
        )

    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program,
                               evaluation_feedback: Optional[Dict[str, Any]] = None,
                               ancestral_summary: Optional[
                                   List[Dict[str, Any]]] = None) -> str:  # Method_v2 (added ancestral_summary)
        logger.info(
            f"Designing mutation prompt for program: {parent_program.id} (Gen: {parent_program.generation}, Mode: {task.improvement_mode})")
        feedback_summary_str = self._format_evaluation_feedback(task, parent_program, evaluation_feedback)
        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)

        logger.debug(f"Formatted evaluation feedback for prompt:\n{feedback_summary_str}")
        if historical_context_section:
            logger.debug(f"Ancestral summary for prompt:\n{historical_context_section}")

        # Standard diff instructions, applicable to both modes
        diff_instructions = (
            "Your Response Format:\n"
            "Propose improvements to the 'Current Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Current Code'. Do not paraphrase or shorten it."
            "- If you are adding new code where nothing existed, the SEARCH block can be a comment indicating the location, or an adjacent existing line."
            "- If you are deleting code, the REPLACE block should be empty."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )
        prompt_header = f"You are an expert Python programmer. Your task is to improve an existing Python function based on its previous performance, historical attempts, and the overall goal.\n\n" # Added "historical attempts"
        current_code_section = (
            f"Current Code (Version from Generation {parent_program.generation}):\n"
            f"```python\n{parent_program.code}\n```\n\n"
            f"Evaluation Feedback on the 'Current Code':\n{feedback_summary_str}\n\n"
        )
        allowed_imports_section = f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use other external libraries or packages.\n\n" # Use task, not self.task_definition here

        # --- Start of new logic for different improvement modes (v2) ---
        if self.task_definition.improvement_mode == "general_refinement":
            logger.info(f"Crafting mutation prompt for 'general_refinement' mode.")
            improvement_goal_intro = "Your Improvement Goal (General Refinement):\n"

            directives_section = ""
            if self.task_definition.specific_improvement_directives:
                directives_section = f"Specific Directives: {self.task_definition.specific_improvement_directives}\n"

            metrics_section = ""
            if self.task_definition.primary_focus_metrics:
                metrics_str = ", ".join(self.task_definition.primary_focus_metrics)
                metrics_section = f"Primary Focus Metrics: Aim to improve metrics such as {metrics_str}. (Note: Detailed scores for these may not be in current feedback but will be evaluated on your new version).\n"
            else:
                metrics_section = "Primary Focus Metrics: Aim to improve general code quality (e.g., readability, maintainability, efficiency where applicable).\n"

            improvement_goal_details = (
                f"Based on the 'Current Code', its 'Evaluation Feedback', and the 'Brief History...', your goal is to propose modifications to refine and enhance this code. "
                f"{directives_section}"
                f"{metrics_section}"
                f"If input/output examples were provided with the original task/seed, ensure your changes do not break existing functionality (treat I/O feedback as regression tests).\n\n"
            )
            prompt = (
                prompt_header +
                f"Overall Task Context (if any): {task.description}\n\n" +
                f"Function of Interest: `{task.function_name_to_evolve}`\n\n" +
                allowed_imports_section +
                current_code_section +
                historical_context_section + # <-- Inserted history
                improvement_goal_intro +
                improvement_goal_details +
                diff_instructions
            )

        else:  # Default to "task_focused" mode (original behavior with slight refinement)
            logger.info(f"Crafting mutation prompt for 'task_focused' mode.")
            improvement_goal_intro = "Your Improvement Goal (Task Focused):\n"
            improvement_goal_details = (
                f"Based on the task, the 'Current Code', its 'Evaluation Feedback', and the 'Brief History...', your goal is to propose modifications to improve the function `{task.function_name_to_evolve}` "
                f"to better solve the defined task: {task.description}\n"
                f"Prioritize fixing any errors or correctness issues."
                f"If correct, focus on efficiency or alternative robust logic."
                f"Consider the original evaluation criteria: {task.evaluation_criteria}\n\n"
            )
            prompt = (
                prompt_header +
                f"Overall Task Description: {task.description}\n\n" +
                f"Function to Improve: `{task.function_name_to_evolve}`\n\n" +
                allowed_imports_section +
                current_code_section +
                historical_context_section + # <-- Inserted history
                improvement_goal_intro +
                improvement_goal_details +
                diff_instructions
            )
        # --- End of new logic (v2) ---

        logger.debug(
            f"Designed mutation prompt (mode: {self.task_definition.improvement_mode}, requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_message: str,
                              execution_output: Optional[str] = None,
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str: # Method_v2 (added ancestral_summary)
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Gen: {program.generation}, Mode: {task.improvement_mode})")
        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        if historical_context_section:
            logger.debug(f"Ancestral summary for bug-fix prompt:\n{historical_context_section}")

        output_segment = f"Execution Output (stdout/stderr that might be relevant):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the error message itself.\n"

        diff_instructions = (
            "Your Response Format:\n"
            "Propose fixes to the 'Buggy Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Buggy Code'."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to fix a bug in an existing Python function, considering previous attempts.\n\n" # Added "considering previous attempts"
            f"Overall Task Description: {task.description}\n"
            f"Function to Fix: `{task.function_name_to_evolve}`\n"
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Buggy Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Error Encountered: {error_message}\n"
            f"{output_segment}"
            f"{historical_context_section}" # <-- Inserted history
            f"Your Goal:\n"
            f"Analyze the 'Buggy Code', 'Error Encountered', 'Execution Output', and the 'Brief History...' to identify and fix the bug(s). "
            f"The corrected function must adhere to the overall task description and allowed imports.\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_failed_diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,
                                           ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str: # Method_v2 (added ancestral_summary)
        logger.info(f"Designing failed-diff fallback prompt for program: {original_program.id} (Gen: {original_program.generation})")
        historical_context_section = self._format_ancestral_summary_for_prompt(ancestral_summary)
        if historical_context_section:
            logger.debug(f"Ancestral summary for fallback prompt:\n{historical_context_section}")

        prompt = (
            f"You are an expert Python programmer. Your previous attempt to provide improvements for the following code via a diff format was unsuccessful or resulted in no changes.\n\n"
            f"Overall Task Description: {task.description}\n"
            f"Function of Interest: `{task.function_name_to_evolve}`\n"
            f"Allowed Standard Library Imports: {task.allowed_imports}. Do not use any other external libraries or packages.\n\n"
            f"Original Code (that was attempted to be diffed):\n"
            f"```python\n{original_program.code}\n```\n\n"
            f"Summary of Previous Attempt/Goal: {previous_attempt_summary}\n\n"
            f"{historical_context_section}" # <-- Inserted history
            f"Your New Goal:\n"
            f"Please provide the *complete, fully corrected/improved version* of the function `{task.function_name_to_evolve}`. "
            f"Ensure your new version addresses the objectives mentioned in the 'Summary of Previous Attempt/Goal' and learns from any 'Brief History...'.\n\n" # Added learns from history
            f"Your Response Format:\n"
            f"Provide *only* the complete Python code for the new version of the function `{task.function_name_to_evolve}`. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences."
        )
        logger.debug(f"Designed failed-diff fallback prompt:\n{prompt}")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PromptDesignerAgent.execute() is not the primary way to use this agent. Call specific design methods.")