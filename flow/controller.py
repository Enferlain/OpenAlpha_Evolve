# task_manager/controller.py
# Version: 1.1.0 (Adding LLM Review Score calculation for monitoring)

import logging
import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Union

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, PromptDesignerInterface, CodeGeneratorInterface, SolutionEvaluatorInterface,
    DatabaseInterface, SelectionControllerInterface,
    MetricsLoggerInterface
)
from config import settings
from engine.prompting import PromptStudio
from engine.generation import CodeProducer
# Ensure SolutionEvaluator is imported if its type hint is used, or adjust if only interface used.
from engine.evaluation import SolutionEvaluator
from flow.database_sqlite import SQLiteStore
from flow.selection import EvoSelector
from flow.monitoring import MetricsLogger

logger = logging.getLogger(__name__)

SELECTED_MODEL_FOR_GENERATION = settings.GENERATION_MODEL_NAME
RPM_LIMIT = settings.MODEL_FREE_TIER_RPM.get(
    SELECTED_MODEL_FOR_GENERATION,
    settings.MODEL_FREE_TIER_RPM["default"]
)
MAX_CONCURRENT_API_CALLS = 1
MIN_SECONDS_BETWEEN_CALLS = 60.0 / RPM_LIMIT if RPM_LIMIT > 0 else 6.0

logger.info(
    f"API Call Management for Model '{SELECTED_MODEL_FOR_GENERATION}': "
    f"RPM_LIMIT={RPM_LIMIT}, MAX_CONCURRENT_API_CALLS={MAX_CONCURRENT_API_CALLS}, "
    f"MIN_SECONDS_BETWEEN_CALLS={MIN_SECONDS_BETWEEN_CALLS:.2f}s"
)


class EvolveFlow(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition

        # Initialize agents, ensuring SolutionEvaluator gets PromptDesigner and CodeGenerator for ai calls
        self.prompt_designer: PromptDesignerInterface = PromptStudio(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeProducer()

        # Pass the necessary agents to SolutionEvaluator for LLM-as-reviewer
        self.evaluator: SolutionEvaluatorInterface = SolutionEvaluator(
            task_definition=self.task_definition,
            prompt_designer=self.prompt_designer,  # For designing ai review prompts
            code_generator_for_=self.code_generator  # For making the call to the LLM
        )

        self.database: DatabaseInterface = SQLiteStore()
        self.selection_controller: SelectionControllerInterface = EvoSelector()
        self.monitoring_agent: MetricsLoggerInterface = MetricsLogger()

        self.crossover_rate = settings.CROSSOVER_RATE
        self.min_parents_for_crossover = 2

        self._api_call_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
        self._last_api_call_release_time = 0.0

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        self.num_parents_to_select = self.population_size // 2
        if self.num_parents_to_select < settings.ELITISM_COUNT and self.population_size >= settings.ELITISM_COUNT:
            self.num_parents_to_select = settings.ELITISM_COUNT
        elif self.num_parents_to_select == 0 and self.population_size > 0:
            self.num_parents_to_select = 1

        self.start_time_overall_run: Optional[float] = None

    async def initialize_population(self) -> List[Program]:  # Method_v2.1.1 (Using initial_seed)
        logger.info(
            f"Initializing population for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        initial_population = []
        programs_to_generate_from_prompt = self.population_size
        start_index_for_prompt_generation = 0

        # --- CORRECTED to use initial_seed ---
        if self.task_definition.initial_seed and \
                self.task_definition.improvement_mode == "general_refinement":
            logger.info(f"Using provided 'initial_seed' for the first program.")
            program_id = f"{self.task_definition.id}_gen0_seed_prog0"

            # If it's just "ideas" (text), it might not be valid code directly.
            # For now, we'll assume if it's provided, it's intended as the 'code' field.
            # The PromptStudio for initial prompt also uses this field.
            # If it's ideas, the initial program might fail syntax but that's okay,
            # as it's a seed for refinement.
            seed_content = self.task_definition.initial_seed

            seed_program = Program(
                id=program_id,
                code=seed_content if seed_content else "# Empty seed provided via initial_seed",
                generation=0,
                status="unevaluated",  # Will be evaluated
                task_id=self.task_definition.id,
                creation_method="initial_seed"  # This creation_method is fine
            )
            initial_population.append(seed_program)
            programs_to_generate_from_prompt -= 1
            start_index_for_prompt_generation = 1
            logger.info(f"Will generate {programs_to_generate_from_prompt} additional programs using initial prompt.")

        if programs_to_generate_from_prompt > 0:
            # --- CORRECTED to use initial_seed ---
            if self.task_definition.improvement_mode == "general_refinement" and \
                    not self.task_definition.initial_seed:
                logger.warning(
                    f"Task mode is 'general_refinement' but no 'initial_seed' was provided. "
                    f"Proceeding with standard initial prompt for task '{self.task_definition.id}'.")

            # The PromptStudio's initial_prompt is already updated to use initial_seed
            initial_prompt_text = self.prompt_designer.initial_prompt(self.task_definition)

            # Safety check if prompt is empty (PromptDesigner should handle this, but defense in depth)
            if not initial_prompt_text.strip():
                logger.error(
                    f"Initial prompt for task {self.task_definition.id} was empty. Cannot generate initial population from prompt.")
                # Potentially return empty list or raise error, for now, just won't generate.
            else:
                for i in range(start_index_for_prompt_generation, self.population_size):
                    if len(initial_population) >= self.population_size:
                        break
                    program_id = f"{self.task_definition.id}_gen0_prog{i}"
                    logger.debug(
                        f"Generating initial program {len(initial_population) + 1}/{self.population_size} (ID: {program_id}) using prompt.")

                    async with self._api_call_semaphore:
                        current_time = time.monotonic()
                        time_since_last_call = current_time - self._last_api_call_release_time
                        if time_since_last_call < MIN_SECONDS_BETWEEN_CALLS:
                            sleep_duration = MIN_SECONDS_BETWEEN_CALLS - time_since_last_call
                            logger.debug(f"Rate limiting: sleeping for {sleep_duration:.2f}s before next API call.")
                            await asyncio.sleep(sleep_duration)

                        generated_code = await self.code_generator.generate_code(initial_prompt_text,
                                                                                 temperature=settings.TEMPERATURE_INITIAL_GEN)
                        self._last_api_call_release_time = time.monotonic()

                    program = Program(
                        id=program_id,
                        code=generated_code,
                        generation=0,
                        status="unevaluated",
                        task_id=self.task_definition.id,
                        creation_method="initial_prompt"
                    )
                    initial_population.append(program)

        logger.info(f"Initialized population with {len(initial_population)} programs.")
        return initial_population

    # --- MODIFIED: _calculate_population_metrics (Blueprint Step 6 - Final part) ---
    def _calculate_population_metrics(self, population: List[Program]) -> Dict[str, Any]: # Method_v1.2.0
        """Calculates summary statistics for a population, including LLM  scores."""
        if not population:
            return { # Return defaults for all expected metrics by MetricsLogger
                "avg_correctness": 0.0, "best_correctness": 0.0,
                "avg_ruff_violations": float('inf'), "min_ruff_violations": float('inf'),
                "ai_review_score_avg": settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0), # NEW
                "ai_review_score_best": settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0), # NEW
                "avg_runtime_ms": float('inf'), "best_runtime_ms": float('inf'),
                "avg_cyclomatic_complexity": float('inf'), "best_cyclomatic_complexity": float('inf'),
                "avg_maintainability_index": settings.DEFAULT_METRIC_VALUE.get("maintainability_index", 0.0),
                "best_maintainability_index": settings.DEFAULT_METRIC_VALUE.get("maintainability_index", 0.0),
            }

        metrics_data = {
            "correctness": [],
            "ruff_violations": [],
            "ai_review_score": [], # NEW list for ai review scores
            "runtime_ms": [],
            "cyclomatic_complexity_avg": [],
            "maintainability_index": []
        }

        evaluated_programs = [p for p in population if p.status == "evaluated" and p.fitness_scores]

        for prog in evaluated_programs:
            metrics_data["correctness"].append(prog.fitness_scores.get("correctness", 0.0))
            metrics_data["ruff_violations"].append(prog.fitness_scores.get("ruff_violations", float('inf')))
            # --- NEW: Collect ai review Scores ---
            metrics_data["ai_review_score"].append(
                prog.fitness_scores.get("ai_review_score", settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0))
            )
            # --- END NEW ---
            metrics_data["runtime_ms"].append(prog.fitness_scores.get("runtime_ms", float('inf')))
            metrics_data["cyclomatic_complexity_avg"].append(
                prog.fitness_scores.get("cyclomatic_complexity_avg", float('inf')))
            metrics_data["maintainability_index"].append(prog.fitness_scores.get("maintainability_index",
                                                                            settings.DEFAULT_METRIC_VALUE.get("maintainability_index", 0.0)))

        def safe_avg(values: List[Union[float, int]], default_val=0.0) -> float:
            valid_values = [v for v in values if isinstance(v, (int, float)) and v != float('inf') and v != float('-inf') and not (isinstance(v, float) and v != v)]
            return sum(valid_values) / len(valid_values) if valid_values else default_val

        def safe_min(values: List[Union[float, int]], default_val=float('inf')) -> float:
            valid_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)]
            if not valid_values: return default_val
            return min(valid_values) if any(v != float('inf') for v in valid_values) else default_val

        def safe_max(values: List[Union[float, int]], default_val=0.0) -> float:
            valid_values = [v for v in values if isinstance(v, (int, float)) and v != float('-inf') and not (isinstance(v, float) and v != v)]
            if not valid_values: return default_val
            return max(valid_values) if any(v != float('-inf') for v in valid_values) else default_val

        summary = {
            "avg_correctness": safe_avg(metrics_data["correctness"]),
            "best_correctness": safe_max(metrics_data["correctness"]),
            "avg_ruff_violations": safe_avg(metrics_data["ruff_violations"], default_val=float('inf')),
            "min_ruff_violations": safe_min(metrics_data["ruff_violations"]),
            # --- NEW: Calculate Avg and Best ai review Scores ---
            "ai_review_score_avg": safe_avg(metrics_data["ai_review_score"], default_val=settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0)),
            "ai_review_score_best": safe_max(metrics_data["ai_review_score"], default_val=settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0)),
            # --- END NEW ---
            "avg_runtime_ms": safe_avg(metrics_data["runtime_ms"], default_val=float('inf')),
            "best_runtime_ms": safe_min(metrics_data["runtime_ms"]),
            "avg_cyclomatic_complexity": safe_avg(metrics_data["cyclomatic_complexity_avg"], default_val=float('inf')),
            "best_cyclomatic_complexity": safe_min(metrics_data["cyclomatic_complexity_avg"]),
            "avg_maintainability_index": safe_avg(metrics_data["maintainability_index"], default_val=settings.DEFAULT_METRIC_VALUE.get("maintainability_index", 0.0)),
            "best_maintainability_index": safe_max(metrics_data["maintainability_index"], default_val=settings.DEFAULT_METRIC_VALUE.get("maintainability_index", 0.0)),
        }
        logger.debug(f"Calculated population metrics: {summary}")
        return summary

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs_accumulator = []  # Renamed to avoid confusion with the input 'population'

        # Create evaluation tasks only for programs that haven't been evaluated yet
        # or need re-evaluation (though current logic doesn't explicitly re-evaluate)
        evaluation_tasks = []
        programs_to_evaluate_indices = []  # Keep track of original indices

        for i, prog in enumerate(population):
            if prog is not None and prog.status != "evaluated":  # Make sure prog isn't None here too
                evaluation_tasks.append(self.evaluator.evaluate_program(prog, self.task_definition))
                programs_to_evaluate_indices.append(i)
            elif prog is not None:  # Already evaluated, just add it back
                evaluated_programs_accumulator.append(prog)
            # If prog is None initially, we skip it, it shouldn't be in the list.

        if not evaluation_tasks:
            logger.info("No programs in the provided list needed evaluation.")
            # Ensure we return a list of Programs, even if some were None initially in the input
            return [p for p in population if isinstance(p, Program)]

            # --- This part processes results from asyncio.gather ---
        results_from_gather = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Create a temporary list for newly evaluated programs
        newly_evaluated_or_failed_programs = []

        for i, gather_result in enumerate(results_from_gather):
            original_program_index = programs_to_evaluate_indices[i]  # Get the original index
            program_being_processed = population[original_program_index]  # Get the Program object sent for evaluation

            if isinstance(gather_result, Exception):
                logger.error(f"Error evaluating program {program_being_processed.id}: {gather_result}",
                             exc_info=gather_result)
                program_being_processed.status = "failed_evaluation"
                program_being_processed.errors.append(
                    f"AsyncGather Exception: {type(gather_result).__name__} - {str(gather_result)}")
                newly_evaluated_or_failed_programs.append(program_being_processed)
            elif isinstance(gather_result, Program):  # Explicitly check if it's a Program object
                newly_evaluated_or_failed_programs.append(gather_result)
            else:
                # This case means evaluate_program returned something unexpected (like None)
                # that wasn't an exception. This is the problem area.
                logger.error(
                    f"SolutionEvaluator.evaluate_program for program {program_being_processed.id} returned an unexpected type: {type(gather_result)}. Value: {gather_result}. Marking as failed.")
                program_being_processed.status = "failed_evaluation_internal"  # A distinct status
                program_being_processed.errors.append(f"Evaluator returned unexpected type: {type(gather_result)}")
                newly_evaluated_or_failed_programs.append(
                    program_being_processed)  # Add the original program in its failed state

            # Save to database immediately after processing each result
            program_to_save = newly_evaluated_or_failed_programs[-1]
            if isinstance(program_to_save, Program):  # Ensure it's a Program before saving
                await self.database.save_program(program_to_save)
            else:
                # This should ideally not be reached if the above logic is correct
                logger.critical(
                    f"Attempted to save a non-Program object ({type(program_to_save)}) to database. This indicates a logic flaw. Program ID was: {program_being_processed.id if program_being_processed else 'Unknown'}")

        # Reconstruct the full list of programs in the original order if necessary,
        # or simply combine already_evaluated with newly_evaluated.
        # For simplicity, let's just combine. The order might change if that matters for selection.
        # If order from input `population` must be strictly preserved, more complex merging is needed.

        final_evaluated_population = []
        # Add back programs that were already evaluated and skipped
        for prog in population:
            if prog is not None and prog.status == "evaluated" and prog not in newly_evaluated_or_failed_programs:  # Avoid duplicates if they were re-evaluated
                final_evaluated_population.append(prog)

        final_evaluated_population.extend(p for p in newly_evaluated_or_failed_programs if isinstance(p, Program))

        logger.info(f"Finished evaluating. Total programs processed/retained: {len(final_evaluated_population)}.")
        return final_evaluated_population

    async def run_evolution(self) -> List[Program]:  # Method_v6.1 (minor logging for total API calls)
        logger.info(
            f"Starting evolutionary cycle for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        self.start_time_overall_run = time.monotonic()

        if isinstance(self.database, SQLiteStore):
            await self.database.setup_db()

        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            start_time_generation = time.monotonic()
            if hasattr(self.code_generator, 'reset_api_call_count_generation'):
                self.code_generator.reset_api_call_count_generation()
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            parents = self.selection_controller.get_parents(current_population, self.num_parents_to_select,
                                                               self.task_definition)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break

            offspring_population: List[Program] = []
            num_newly_generated_offspring = 0
            generation_tasks = []
            for i in range(self.population_size):  # Assuming we generate population_size new offspring
                child_id_base = f"{self.task_definition.id}_gen{gen}"
                if len(parents) >= self.min_parents_for_crossover and random.random() < self.crossover_rate:
                    if len(parents) < 2:  # Should not happen if num_parents_to_select >= 2 for crossover
                        p1 = random.choice(parents)
                        generation_tasks.append(self.generate_offspring(p1, gen, f"{child_id_base}_mut{i}"))
                    else:
                        p1, p2 = random.sample(parents, 2)
                        generation_tasks.append(
                            self.generate_crossover_offspring(p1, p2, gen, f"{child_id_base}_cross{i}"))
                else:
                    p1 = random.choice(parents)
                    generation_tasks.append(self.generate_offspring(p1, gen, f"{child_id_base}_mut{i}"))

            if generation_tasks:
                generated_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
                for result in generated_results:
                    if isinstance(result, Program):
                        offspring_population.append(result)
                        await self.database.save_program(result)  # Save offspring before evaluation
                        num_newly_generated_offspring += 1
                    elif isinstance(result, Exception):
                        logger.error(f"Error during offspring generation task: {result}", exc_info=result)

            if offspring_population:
                offspring_population = await self.evaluate_population(offspring_population)

            current_population = self.selection_controller.get_survivors(current_population, offspring_population,
                                                                            self.population_size, self.task_definition)

            # Log best program of this generation
            if current_population:
                sorted_for_log = self.selection_controller.sort_programs(current_population, self.task_definition)
                if sorted_for_log: logger.info(
                    f"Generation {gen}: Best program in new pop: ID={sorted_for_log[0].id}, Fitness={sorted_for_log[0].fitness_scores}")

            generation_time_sec = time.monotonic() - start_time_generation
            population_metrics = self._calculate_population_metrics(current_population)  # This now has Ruff metrics
            llm_calls_this_generation = self.code_generator.get_api_call_count_generation() if hasattr(
                self.code_generator, 'get_api_call_count_generation') else 0

            generation_log_data = {
                "task_id": self.task_definition.id, "generation_number": gen,
                "population_size": len(current_population), "num_offspring_generated": num_newly_generated_offspring,
                **population_metrics, "generation_time_sec": round(generation_time_sec, 2),
                "llm_api_calls_generation": llm_calls_this_generation
            }
            await self.monitoring_agent.execute("log_generation_metrics", payload=generation_log_data)

            if not current_population: logger.warning(
                f"Generation {gen}: No programs in current population. Ending evolution."); break

        logger.info("Evolutionary cycle completed.")
        total_run_time_sec = time.monotonic() - (
            self.start_time_overall_run if self.start_time_overall_run else time.monotonic())
        total_api_calls_session = self.code_generator.get_api_call_count_session() if hasattr(self.code_generator,
                                                                                              'get_api_call_count_session') else 0

        best_program_list = await self._get_best_program()
        best_overall_program_obj = best_program_list[0] if best_program_list else None

        final_summary_payload = {
            "best_program_overall": best_overall_program_obj, "total_runtime_sec": round(total_run_time_sec, 2),
            "task_id": self.task_definition.id, "total_llm_api_calls_session": total_api_calls_session
        }
        await self.monitoring_agent.execute("log_run_summary", payload=final_summary_payload)
        return best_program_list

    async def generate_crossover_offspring(self, parent1: Program, parent2: Program, generation_num: int,
                                           child_id: str) -> Optional[Program]:  # v1.0.0
        logger.debug(
            f"Generating crossover offspring {child_id} from parents {parent1.id} & {parent2.id} for gen {generation_num}")

        crossover_prompt = self.prompt_designer.crossover_prompt(self.task_definition, parent1, parent2)

        # For crossover, we expect a full new code block
        generated_code = await self.code_generator.generate_code(  # Use generate_code directly
            prompt=crossover_prompt,
            temperature=settings.TEMPERATURE_CROSSOVER,  # May want a slightly different temperature for creative synthesis
            output_format="code"  # Expecting complete code
        )

        if not generated_code.strip():
            logger.warning(f"Crossover for {child_id} resulted in empty code. Skipping.")
            return None

        # Simple check: if returned code is identical to one of the parents, maybe it wasn't a good crossover
        if generated_code == parent1.code or generated_code == parent2.code:
            logger.info(f"Crossover for {child_id} resulted in code identical to one parent. Still proceeding.")
            # We might decide to penalize this or retry later, but for now, accept it.

        offspring = Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_ids=[parent1.id, parent2.id],  # Store both parent IDs!
            status="unevaluated",
            task_id=self.task_definition.id,
            creation_method="crossover"  # Track how it was made!
        )
        logger.info(f"Successfully generated crossover offspring {offspring.id}.")
        return offspring

    async def generate_offspring(self, parent: Program, generation_num: int, child_id: str) -> Optional[
        Program]:  # Method_v4.2 (Corrected bug_fix_prompt call)
        logger.debug(f"Generating mutation/bug_fix offspring {child_id} from parent {parent.id}")
        ancestral_summary = await self._get_ancestral_summary(parent, max_depth=3)

        mutation_prompt_str: str = ""
        prompt_type: str = "mutation"
        original_attempt_summary_for_fallback = ""

        # Decide if bug_fix or mutation based on status and errors
        # The PromptStudio.bugfix_prompt will internally look at program.status and program.errors
        # to determine the "Primary Problem".

        # Heuristic: If status indicates a clear failure (not just failed I/O tests but execution/syntax)
        # or if ai review gave a very low score indicating critical flaws.
        is_critical_failure_status = parent.status in ["failed_evaluation_execution", "failed_evaluation_syntax",
                                                       "failed_evaluation_internal_critical"]
        is_very_low_judge_score = parent.fitness_scores.get('ai_review_score', 10) <= 3  # Example threshold

        if is_critical_failure_status or (parent.ai_review_feedback and is_very_low_judge_score):
            logger.info(
                f"Attempting bug-fix prompt for {parent.id} due to status '{parent.status}' or low judge score.")
            # --- CORRECTED CALL: Removed error_message and execution_output ---
            mutation_prompt_str = self.prompt_designer.bugfix_prompt(
                task=self.task_definition,
                program=parent,  # PromptDesigner will get error context from here
                ancestral_summary=ancestral_summary
            )
            # --- END CORRECTION ---
            prompt_type = "bug_fix"
            # The original_attempt_summary_for_fallback can be generic or try to infer from parent.errors
            primary_error_display = parent.errors if parent.errors else "an issue"
            original_attempt_summary_for_fallback = f"Fix critical issue (e.g., '{primary_error_display[:50]}...') for task: {self.task_definition.description}"
        else:
            logger.info(f"Attempting mutation prompt for {parent.id}.")
            mutation_prompt_str = self.prompt_designer.mutation_prompt(
                task=self.task_definition,
                parent_program=parent,
                ancestral_summary=ancestral_summary
            )
            prompt_type = "mutation"
            original_attempt_summary_for_fallback = f"Improve code for task: {self.task_definition.description}, mode: {self.task_definition.improvement_mode}"

        final_code = ""
        # --- Rate limiting and API call logic for diff/fallback ---
        # (Assuming this block is okay, as the error was in the prompt design call)
        async with self._api_call_semaphore:
            current_time = time.monotonic()
            if current_time - self._last_api_call_release_time < MIN_SECONDS_BETWEEN_CALLS:
                await asyncio.sleep(MIN_SECONDS_BETWEEN_CALLS - (current_time - self._last_api_call_release_time))

            # First attempt: try to get a diff if it's not a brand new attempt after critical failure
            # or if the prompt designer didn't already ask for full code.
            # For simplicity, let's assume the prompt_type from design_x_prompt could indicate if diff is preferred.
            # Our current mutation/bug_fix prompts *do* ask for diffs.

            temp_generated_output = await self.code_generator.execute(  # Using .execute now
                prompt=mutation_prompt_str,
                temperature=settings.TEMPERATURE_MUTATION_DIFF,
                output_format="diff",  # Requesting diff
                parent_code_for_diff=parent.code
            )
            self._last_api_call_release_time = time.monotonic()

        # Check if the output is actually a diff or if it's already full code (e.g. LLM ignored diff request)
        # or if the diff application failed or resulted in no change.
        # The CodeProducer's execute method with output_format="diff" returns the *applied* code.
        # If it couldn't apply or diff was bad, it might return original or raw diff.
        # Let's simplify: if the result from a diff request is same as parent or empty, or still looks like a diff, try fallback.

        # The CodeProducer's execute for "diff" already tries to apply it.
        # If it returns the parent code, it means the diff failed or was empty.
        if not temp_generated_output.strip() or temp_generated_output == parent.code:
            logger.warning(
                f"Diff attempt for {child_id} ({prompt_type}) failed or no change. Trying fallback to full code generation.")
            prompt_type_original = prompt_type
            prompt_type += "_fallback_full"  # Mark that fallback was used

            fallback_prompt = self.prompt_designer.diff_fallback_prompt(
                task=self.task_definition, original_program=parent,
                previous_attempt_summary=original_attempt_summary_for_fallback,
                # Pass summary of what we tried to fix/improve
                ancestral_summary=ancestral_summary
            )
            async with self._api_call_semaphore:  # API call rate limiting for fallback
                current_time = time.monotonic()
                if current_time - self._last_api_call_release_time < MIN_SECONDS_BETWEEN_CALLS:
                    await asyncio.sleep(MIN_SECONDS_BETWEEN_CALLS - (current_time - self._last_api_call_release_time))
                final_code = await self.code_generator.execute(  # Using .execute
                    prompt=fallback_prompt,
                    temperature=settings.TEMPERATURE_FALLBACK_FULL,
                    output_format="code"  # Requesting full code
                )
                self._last_api_call_release_time = time.monotonic()

            if not final_code.strip() or final_code == parent.code:
                logger.warning(
                    f"Fallback full code generation for {child_id} ({prompt_type_original}) also empty or no change. Skipping offspring generation.")
                return None
        else:  # Diff was presumably successful and applied by CodeProducer.execute
            final_code = temp_generated_output

        offspring = Program(id=child_id, code=final_code, generation=generation_num, parent_id=parent.id,
                            parent_ids=[parent.id] if parent.id else [], status="unevaluated",
                            task_id=self.task_definition.id, creation_method=prompt_type)
        return offspring

    async def _get_best_program(self) -> List[Program]:  # Helper method extracted
        logger.info("Fetching overall best program from database...")
        all_evaluated_programs = await self.database.get_all_programs()

        candidate_programs = [p for p in all_evaluated_programs if
                              p.status == "evaluated" and p.task_id == self.task_definition.id]  # Filter by task_id

        if not candidate_programs:
            logger.info(f"No evaluated programs found for task {self.task_definition.id} to determine an overall best.")
            return []

        final_best_sorted = self.selection_controller.sort_programs(candidate_programs, self.task_definition)
        if not final_best_sorted:
            logger.info(f"Sorting of evaluated programs for task {self.task_definition.id} yielded no result.")
            return []

        best_overall_program = final_best_sorted[0]
        logger.info(
            f"Overall Best Program Found for task '{self.task_definition.id}': ID={best_overall_program.id}, Gen={best_overall_program.generation}, Method='{best_overall_program.creation_method}', Fitness={best_overall_program.fitness_scores}"
        )
        # For detailed display, could log code here or just return it.
        # logger.info(f"Code:\n{best_overall_program.code}")
        return [best_overall_program]

    # --- MODIFIED: _get_ancestral_summary (v1.1.0 for Ruff) ---
    async def _get_ancestral_summary(self, program: Program, max_depth: int = 3) -> List[
        Dict[str, Any]]:  # Method_v1.1.0
        """
        Retrieves a concise summary of a program's recent ancestors for LLM prompting.
        Includes Ruff violation counts in the outcome summary.
        """
        logger.debug(f"Fetching ancestral summary for program {program.id}, max_depth={max_depth}.")
        history_summary = []
        current_prog_in_history = program
        processed_ids = {program.id}

        for _ in range(max_depth):
            parent_ref_id = current_prog_in_history.parent_id or \
                            (current_prog_in_history.parent_ids[0] if current_prog_in_history.parent_ids else None)

            if not parent_ref_id or parent_ref_id in processed_ids:
                break

            parent = await self.database.get_program(parent_ref_id)
            if not parent:
                logger.warning(f"Could not retrieve parent {parent_ref_id} for ancestral history of {program.id}.")
                break

            processed_ids.add(parent.id)

            outcome_parts = []
            if parent.fitness_scores.get("correctness") is not None:
                outcome_parts.append(f"Correctness: {parent.fitness_scores['correctness'] * 100:.0f}%")

            ruff_v = parent.fitness_scores.get("ruff_violations")
            if ruff_v is not None and ruff_v != float('inf'):
                outcome_parts.append(f"Ruff Violations: {int(ruff_v)}")

            # Summarize errors briefly
            error_summary_text = ""
            if parent.errors:
                # Count execution vs Ruff errors if desired, or just indicate "had issues"
                num_exec_errors = sum(1 for e in parent.errors if not e.lower().startswith("ruff-"))
                if num_exec_errors > 0 and parent.fitness_scores.get("correctness", 1.0) < 1.0:
                    error_summary_text = " (failed I/O/exec)"
                elif parent.errors:  # Any errors left (could be only Ruff)
                    error_summary_text = " (had static issues)"

            outcome_str = ', '.join(outcome_parts) + error_summary_text
            if not outcome_str.strip(): outcome_str = "Prior version (no specific metrics captured in summary)."

            summary_entry = {
                "generation": parent.generation,
                "creation_method": parent.creation_method if parent.creation_method != "unknown" else "prev_gen",
                "outcome_summary": outcome_str
            }
            history_summary.append(summary_entry)
            current_prog_in_history = parent

        logger.debug(f"Returning {len(history_summary)} ancestral summaries for {program.id}.")
        return list(reversed(history_summary))

    async def execute(self) -> Any:  # Unchanged
        return await self.run_evolution()