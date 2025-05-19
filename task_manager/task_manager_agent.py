import logging
import asyncio
import random
import time # For timing generations and total run
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface,
    MonitoringAgentInterface # <--- ADD THIS!
)
from config import settings

# Import concrete agent implementations
from prompt_designer.prompt_designer_agent import PromptDesignerAgent
from code_generator.code_generator_agent import CodeGeneratorAgent
from evaluator_agent.evaluator_agent import EvaluatorAgent
from database_agent.database_agent import InMemoryDatabaseAgent  # Using InMemory for now
from database_agent.sqlite_database_agent import SQLiteDatabaseAgent # New!
from selection_controller.selection_controller_agent import SelectionControllerAgent
from monitoring_agent.monitoring_agent import MonitoringAgent # Import the concrete agent

logger = logging.getLogger(__name__)

# --- Determine RPM and delay based on the currently selected settings.GENERATION_MODEL_NAME ---
# Look up the RPM in our new dictionary from settings
SELECTED_MODEL_FOR_GENERATION = settings.GENERATION_MODEL_NAME
RPM_LIMIT = settings.MODEL_FREE_TIER_RPM.get(
    SELECTED_MODEL_FOR_GENERATION,
    settings.MODEL_FREE_TIER_RPM["default"] # Use the default fallback if model not found
)

MAX_CONCURRENT_API_CALLS = 1 # Still recommend 1 for simplicity with free tiers
MIN_SECONDS_BETWEEN_CALLS = 60.0 / RPM_LIMIT if RPM_LIMIT > 0 else 6.0 # Default to 6s if RPM is 0/invalid (conservative)

logger.info(
    f"API Call Management for Model '{SELECTED_MODEL_FOR_GENERATION}': "
    f"RPM_LIMIT={RPM_LIMIT}, MAX_CONCURRENT_API_CALLS={MAX_CONCURRENT_API_CALLS}, "
    f"MIN_SECONDS_BETWEEN_CALLS={MIN_SECONDS_BETWEEN_CALLS:.2f}s"
)


class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        # self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        self.database: DatabaseAgentInterface = SQLiteDatabaseAgent()  # New!
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()
        self.monitoring_agent: MonitoringAgentInterface = MonitoringAgent() # Initialize MonitoringAgent

        self.crossover_rate = settings.CROSSOVER_RATE # e.g., 0.2 from settings
        self.min_parents_for_crossover = 2 # Typically 2

        self._api_call_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
        self._last_api_call_release_time = 0.0 # Initialize as float

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        # Calculate num_parents_to_select after potential CLI overrides to population_size
        # This should ideally be done just before starting the evolutionary cycle or passed around.
        # For now, this existing logic is fine as CLI overrides happen in main.py before TaskManager init.
        self.num_parents_to_select = self.population_size // 2
        if self.num_parents_to_select < settings.ELITISM_COUNT and self.population_size >= settings.ELITISM_COUNT:
            self.num_parents_to_select = settings.ELITISM_COUNT # Ensure enough parents for elitism if pop allows
        elif self.num_parents_to_select == 0 and self.population_size > 0:
             self.num_parents_to_select = 1 # Need at least one parent if population exists

        self.start_time_overall_run: Optional[float] = None

    async def initialize_population(self) -> List[Program]:  # Method_v2.1.0 (fix for creation_method)
        logger.info(
            f"Initializing population for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        initial_population = []

        programs_to_generate_from_prompt = self.population_size
        start_index_for_prompt_generation = 0

        if self.task_definition.initial_seed_code and self.task_definition.improvement_mode == "general_refinement":
            logger.info(f"Using provided initial_seed_code for the first program.")
            program_id = f"{self.task_definition.id}_gen0_seed_prog0"
            seed_program = Program(
                id=program_id,
                code=self.task_definition.initial_seed_code,
                generation=0,
                status="unevaluated",
                task_id=self.task_definition.id,  # Ensure task_id is set
                creation_method="initial_seed"
            )
            initial_population.append(seed_program)
            # await self.database.save_program(seed_program) # Let's save all at the end or after evaluation

            programs_to_generate_from_prompt -= 1
            start_index_for_prompt_generation = 1
            logger.info(f"Will generate {programs_to_generate_from_prompt} additional programs using initial prompt.")

        if programs_to_generate_from_prompt > 0:
            if self.task_definition.improvement_mode == "general_refinement" and not self.task_definition.initial_seed_code:
                logger.warning(
                    f"Task mode is 'general_refinement' but no initial_seed_code was provided. Proceeding with standard initial prompt for task '{self.task_definition.id}'. This might not be the intended behavior.")

            initial_prompt_text = self.prompt_designer.design_initial_prompt(self.task_definition)
            if not initial_prompt_text.strip() and self.task_definition.improvement_mode == "general_refinement":
                logger.warning(
                    "Initial prompt from PromptDesigner was empty for general_refinement mode. This is unexpected.")
                initial_prompt_text = (
                    f"You are an expert Python programmer. The primary goal is to work with and refine Python code. "
                    f"The current task is '{self.task_definition.description}'. "
                    f"Function name of interest is '{self.task_definition.function_name_to_evolve}'. "
                    f"Allowed imports: {self.task_definition.allowed_imports}. "
                    f"Provide a complete Python solution. No markdown fences or explanations."
                )

            for i in range(start_index_for_prompt_generation, self.population_size):
                if len(initial_population) >= self.population_size:
                    break

                program_id = f"{self.task_definition.id}_gen0_prog{i}"
                logger.debug(
                    f"Generating initial program {len(initial_population) + 1}/{self.population_size} (ID: {program_id}) using prompt.")

                # Small rate limit for initial generation if multiple calls are very fast
                async with self._api_call_semaphore:
                    current_time = time.monotonic()
                    time_since_last_call = current_time - self._last_api_call_release_time
                    if time_since_last_call < MIN_SECONDS_BETWEEN_CALLS:
                        sleep_duration = MIN_SECONDS_BETWEEN_CALLS - time_since_last_call
                        logger.debug(f"Rate limiting: sleeping for {sleep_duration:.2f}s before next API call.")
                        await asyncio.sleep(sleep_duration)

                    generated_code = await self.code_generator.generate_code(initial_prompt_text, temperature=settings.TEMPERATURE_INITIAL_GEN)
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
                # await self.database.save_program(program) # Let's save all at the end or after evaluation

        # Save all initialized programs to the database (moved from inside the loop)
        # This might be better done after evaluation, but for now, let's ensure they are saved before returning.
        # The evaluate_population method will re-save them with updated status and fitness scores.
        # Actually, initialize_population is called, then evaluate_population.
        # evaluate_population saves programs *after* they are evaluated.
        # So, we need to save them here if they need to be in DB before evaluation starts,
        # or rely on evaluate_population to do the first save.
        # Let's check `evaluate_population`'s behavior.
        # `evaluate_population` calls `self.database.save_program(gather_result)` or the modified program.
        # So, programs created in `initialize_population` don't strictly need to be saved here IF
        # they are immediately passed to `evaluate_population`.
        # Let's assume they are: `current_population = await self.initialize_population()`
        #                       `current_population = await self.evaluate_population(current_population)`
        # This is fine. The `creation_method` will be set on the object, and `evaluate_population` will save it.

        logger.info(
            f"Initialized population with {len(initial_population)} programs. Their creation_methods are now set!")
        return initial_population

    def _calculate_population_metrics(self, population: List[Program]) -> Dict[str, Any]:  # Method_v1.0.0 (New helper)
        """Calculates summary statistics for a population."""
        if not population:
            return {
                "avg_correctness": 0.0, "best_correctness": 0.0,
                "avg_pylint_score": 0.0, "best_pylint_score": 0.0,
                "avg_runtime_ms": float('inf'), "best_runtime_ms": float('inf'),
                "avg_cyclomatic_complexity": float('inf'), "best_cyclomatic_complexity": float('inf'),
                "avg_maintainability_index": 0.0, "best_maintainability_index": 0.0,
            }

        metrics = {
            "correctness": [], "pylint_score": [], "runtime_ms": [],
            "cyclomatic_complexity_avg": [], "maintainability_index": []
        }

        for prog in population:
            if prog.status == "evaluated":  # Only consider evaluated programs for metrics
                metrics["correctness"].append(prog.fitness_scores.get("correctness", 0.0))
                metrics["pylint_score"].append(
                    prog.fitness_scores.get("pylint_score", settings.DEFAULT_METRIC_VALUE.get("pylint_score", -1.0)))
                metrics["runtime_ms"].append(prog.fitness_scores.get("runtime_ms", float('inf')))
                metrics["cyclomatic_complexity_avg"].append(
                    prog.fitness_scores.get("cyclomatic_complexity_avg", float('inf')))
                metrics["maintainability_index"].append(prog.fitness_scores.get("maintainability_index",
                                                                                settings.DEFAULT_METRIC_VALUE.get(
                                                                                    "maintainability_index", -1.0)))

        # Helper to safely calculate avg/min/max
        def safe_avg(values, default=0.0):
            valid_values = [v for v in values if v is not None and v != float('inf') and v != float('-inf')]
            return sum(valid_values) / len(valid_values) if valid_values else default

        def safe_min(values, default=float('inf')):  # For "lower is better" metrics
            valid_values = [v for v in values if v is not None]
            return min(valid_values) if valid_values else default

        def safe_max(values, default=0.0):  # For "higher is better" metrics
            valid_values = [v for v in values if v is not None]
            return max(valid_values) if valid_values else default

        summary = {
            "avg_correctness": safe_avg(metrics["correctness"]),
            "best_correctness": safe_max(metrics["correctness"]),
            "avg_pylint_score": safe_avg(metrics["pylint_score"]),
            "best_pylint_score": safe_max(metrics["pylint_score"], default=-10.0),  # Pylint can be negative
            "avg_runtime_ms": safe_avg(metrics["runtime_ms"], default=float('inf')),
            "best_runtime_ms": safe_min(metrics["runtime_ms"]),
            "avg_cyclomatic_complexity": safe_avg(metrics["cyclomatic_complexity_avg"], default=float('inf')),
            "best_cyclomatic_complexity": safe_min(metrics["cyclomatic_complexity_avg"]),
            "avg_maintainability_index": safe_avg(metrics["maintainability_index"]),
            "best_maintainability_index": safe_max(metrics["maintainability_index"]),
        }
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
                    f"EvaluatorAgent.evaluate_program for program {program_being_processed.id} returned an unexpected type: {type(gather_result)}. Value: {gather_result}. Marking as failed.")
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

    async def manage_evolutionary_cycle(self) -> List[Program]:  # Method_v6 (with LLM Call Count Monitoring)
        logger.info(
            f"Starting evolutionary cycle for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        self.start_time_overall_run = time.monotonic()

        if isinstance(self.database, SQLiteDatabaseAgent):
            await self.database.setup_db()

        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            start_time_generation = time.monotonic()
            if hasattr(self.code_generator, 'reset_api_call_count_generation'):
                self.code_generator.reset_api_call_count_generation()

            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            parents = self.selection_controller.select_parents(
                current_population, self.num_parents_to_select, self.task_definition
            )
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents for reproduction.")

            offspring_population: List[Program] = []
            num_offspring_to_generate = self.population_size
            num_newly_generated_offspring = 0  # Initialize counter for successfully generated offspring

            generation_tasks = []
            for i in range(num_offspring_to_generate):
                if len(parents) >= self.min_parents_for_crossover and random.random() < self.crossover_rate:
                    if len(parents) < 2:  # Ensure enough parents for sampling
                        logger.debug("Not enough parents for crossover, defaulting to mutation for this offspring.")
                        parent_for_mutation = random.choice(parents)  # Should not happen if initial check passes
                        child_id = f"{self.task_definition.id}_gen{gen}_mut{i}"
                        generation_tasks.append(self.generate_offspring(parent_for_mutation, gen, child_id))
                    else:
                        p1, p2 = random.sample(parents, 2)
                        child_id = f"{self.task_definition.id}_gen{gen}_cross{i}"
                        generation_tasks.append(self.generate_crossover_offspring(p1, p2, gen, child_id))
                else:
                    parent_for_mutation = random.choice(parents)
                    child_id = f"{self.task_definition.id}_gen{gen}_mut{i}"
                    generation_tasks.append(self.generate_offspring(parent_for_mutation, gen, child_id))

            if generation_tasks:
                generated_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
                for result in generated_results:
                    if isinstance(result, Program):
                        offspring_population.append(result)
                        await self.database.save_program(result)
                        num_newly_generated_offspring += 1  # Increment for successful offspring
                    elif isinstance(result, Exception):
                        logger.error(f"Error during offspring generation task: {result}", exc_info=result)

            logger.info(f"Generation {gen}: Generated {num_newly_generated_offspring} new offspring candidates.")

            if offspring_population:
                offspring_population = await self.evaluate_population(offspring_population)

            current_population = self.selection_controller.select_survivors(
                current_population, offspring_population, self.population_size, self.task_definition
            )
            logger.info(f"Generation {gen}: New population size after survival: {len(current_population)}.")

            # Log best program of this generation using SelectionController's sorting
            if current_population:
                sorted_for_log = self.selection_controller.sort_programs(current_population, self.task_definition)
                if sorted_for_log:
                    best_program_this_gen = sorted_for_log[0]
                    logger.info(
                        f"Generation {gen}: Best program in new pop: ID={best_program_this_gen.id}, Fitness={best_program_this_gen.fitness_scores}")
                else:
                    logger.warning(f"Generation {gen}: Sorting the current population for logging yielded no results.")

            # --- Monitoring Step for this Generation ---
            generation_time_sec = time.monotonic() - start_time_generation
            population_metrics = self._calculate_population_metrics(current_population)  # Helper method to compute stats

            llm_calls_this_generation = 0
            if hasattr(self.code_generator, 'get_api_call_count_generation'):
                llm_calls_this_generation = self.code_generator.get_api_call_count_generation()

            generation_log_data = {
                "task_id": self.task_definition.id,
                "generation_number": gen,
                "population_size": len(current_population),
                "num_offspring_generated": num_newly_generated_offspring,
                **population_metrics,
                "generation_time_sec": round(generation_time_sec, 2),
                "llm_api_calls_generation": llm_calls_this_generation
            }
            await self.monitoring_agent.execute("log_generation_metrics", payload=generation_log_data)
            # --- End Monitoring Step ---

            if not current_population:
                logger.warning(f"Generation {gen}: No programs in current population. Ending evolution.")
                break

        logger.info("Evolutionary cycle completed.")
        total_run_time_sec = time.monotonic() - self.start_time_overall_run

        best_program_list = await self._get_overall_best_program()  # Helper that uses selection_controller.sort_programs
        best_overall_program_obj = best_program_list[0] if best_program_list else None

        total_api_calls_session = 0
        if hasattr(self.code_generator, 'get_api_call_count_session'):
            total_api_calls_session = self.code_generator.get_api_call_count_session()
        logger.info(f"Total LLM API calls for this session: {total_api_calls_session}")

        final_summary_payload = {
            "best_program_overall": best_overall_program_obj,
            "total_runtime_sec": round(total_run_time_sec, 2),
            "task_id": self.task_definition.id,
            "total_llm_api_calls_session": total_api_calls_session
        }
        # Also add this to the MonitoringAgent's final log if desired (e.g. to metrics file)
        # For now, log_final_summary in MonitoringAgent primarily logs to console.
        await self.monitoring_agent.execute("log_final_summary", payload=final_summary_payload)

        return best_program_list

    async def generate_crossover_offspring(self, parent1: Program, parent2: Program, generation_num: int,
                                           child_id: str) -> Optional[Program]:  # v1.0.0
        logger.debug(
            f"Generating crossover offspring {child_id} from parents {parent1.id} & {parent2.id} for gen {generation_num}")

        crossover_prompt = self.prompt_designer.design_crossover_prompt(self.task_definition, parent1, parent2)

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
        Program]:  # Method_v4 (with Ancestral History and Diff Fallback)
        logger.debug(
            f"Generating mutation/bug_fix offspring {child_id} from parent {parent.id} for gen {generation_num}")

        # Fetch ancestral summary to pass to the prompt designer
        ancestral_summary_for_prompt = await self._get_ancestral_summary_for_llm(parent,
                                                                                 max_depth=3)  # Use the new helper

        mutation_prompt_str: str = ""
        prompt_type: str = "mutation"
        parent_feedback_cleaned = {
            "correctness": parent.fitness_scores.get("correctness"),
            "runtime_ms": parent.fitness_scores.get("runtime_ms"),
            "pylint_score": parent.fitness_scores.get("pylint_score"),
            "cyclomatic_complexity_avg": parent.fitness_scores.get("cyclomatic_complexity_avg"),
            "maintainability_index": parent.fitness_scores.get("maintainability_index"),
            "passed_tests": parent.fitness_scores.get("passed_tests"),
            "total_tests": parent.fitness_scores.get("total_tests"),
            "errors": parent.errors,
        }
        parent_feedback_for_prompt = {k: v for k, v in parent_feedback_cleaned.items() if
                                      v is not None or k == "errors"}
        original_attempt_summary_for_fallback = ""

        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < 0.1:
            primary_error = parent.errors[0]
            mutation_prompt_str = self.prompt_designer.design_bug_fix_prompt(
                task=self.task_definition, program=parent, error_message=primary_error,
                execution_output=None, ancestral_summary=ancestral_summary_for_prompt  # <-- Pass history
            )
            prompt_type = "bug_fix"
            original_attempt_summary_for_fallback = f"Fix the bug causing error: '{primary_error}' in the provided code, considering the task: {self.task_definition.description}"
        else:
            mutation_prompt_str = self.prompt_designer.design_mutation_prompt(
                task=self.task_definition, parent_program=parent,
                evaluation_feedback=parent_feedback_for_prompt,
                ancestral_summary=ancestral_summary_for_prompt  # <-- Pass history
            )
            prompt_type = "mutation"
            original_attempt_summary_for_fallback = f"Improve the code based on its evaluation feedback and the general task: {self.task_definition.description}. Refinement mode: {self.task_definition.improvement_mode}."

        logger.info(f"Attempting {prompt_type} for offspring {child_id} using diff format.")
        final_code = await self.code_generator.execute(
            prompt=mutation_prompt_str, temperature=settings.TEMPERATURE_MUTATION_DIFF, output_format="diff", parent_code_for_diff=parent.code
        )

        diff_failed = False
        if not final_code.strip() or final_code == parent.code:
            diff_failed = True
            logger.warning(
                f"Diff attempt for offspring {child_id} ({prompt_type}) resulted in no change or empty code.")
        elif "<<<<<<< SEARCH" in final_code and ">>>>>>> REPLACE" in final_code and len(final_code) < len(
                parent.code) + 300:  # Increased heuristic length
            diff_failed = True
            logger.warning(
                f"Diff attempt for {child_id} ({prompt_type}) seems to have returned raw diff text. Diff application likely failed.")

        if diff_failed:
            logger.info(f"Diff application failed for {child_id}. Attempting fallback to full code regeneration.")
            creation_method_tag = prompt_type  # Store original attempt type
            prompt_type += "_fallback_full"

            fallback_prompt = self.prompt_designer.design_failed_diff_fallback_prompt(
                task=self.task_definition, original_program=parent,
                previous_attempt_summary=original_attempt_summary_for_fallback,
                ancestral_summary=ancestral_summary_for_prompt  # <-- Pass history to fallback prompt too!
            )

            final_code = await self.code_generator.execute(
                prompt=fallback_prompt, temperature=settings.TEMPERATURE_FALLBACK_FULL, output_format="code"
            )

            if not final_code.strip():
                logger.warning(f"Fallback full code generation for {child_id} also resulted in EMPTY code. Skipping.")
                return None
            if final_code == parent.code:
                logger.warning(f"Fallback full code generation for {child_id} resulted in NO CHANGE. Skipping.")
                return None
            logger.info(
                f"Successfully generated full code for {child_id} via fallback from original attempt '{creation_method_tag}'.")

        offspring = Program(
            id=child_id, code=final_code, generation=generation_num, parent_id=parent.id,
            parent_ids=[parent.id] if parent.id else [], status="unevaluated",
            task_id=self.task_definition.id, creation_method=prompt_type
        )
        logger.info(f"Successfully generated offspring {offspring.id} (Method: {prompt_type}).")
        return offspring

    async def _get_overall_best_program(self) -> List[Program]:  # Helper method extracted
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

    async def _get_ancestral_summary_for_llm(self, program: Program, max_depth: int = 3) -> List[
        Dict[str, Any]]:  # Method_v1.0.0 (New helper for LLM history)
        """
        Retrieves a concise summary of a program's recent ancestors for LLM prompting.
        Returns a list of dictionaries, with the oldest relevant ancestor first.
        """
        logger.debug(f"Fetching ancestral summary for program {program.id}, max_depth={max_depth}.")
        history_summary = []
        current_prog_in_history = program
        # Keep track of processed IDs to prevent loops, though our linear parent_id shouldn't cause them.
        processed_ids = {program.id}

        for i in range(max_depth):
            parent_ref_id = None
            # For simplicity, we trace a single line of ancestry.
            # If a program came from crossover, we'll pick the first parent listed in parent_ids.
            if current_prog_in_history.parent_id:
                parent_ref_id = current_prog_in_history.parent_id
            elif current_prog_in_history.parent_ids and len(current_prog_in_history.parent_ids) > 0:
                parent_ref_id = current_prog_in_history.parent_ids[0]

            if not parent_ref_id or parent_ref_id in processed_ids:
                logger.debug(
                    f"Stopping ancestral search for {program.id}: no more valid parent_ref_id ({parent_ref_id}) or already processed.")
                break

            parent = await self.database.get_program(parent_ref_id)
            if not parent:
                logger.warning(
                    f"Could not retrieve parent {parent_ref_id} for ancestral history of {program.id}. Stopping history trace here.")
                break

            processed_ids.add(parent.id)

            # Create a concise summary for the LLM
            outcome_parts = []
            if parent.fitness_scores.get("correctness") is not None:
                outcome_parts.append(f"Correctness: {parent.fitness_scores['correctness'] * 100:.0f}%")
            if parent.fitness_scores.get("pylint_score") is not None:  # Assuming pylint_score is a float
                p_score = parent.fitness_scores['pylint_score']
                outcome_parts.append(
                    f"Pylint: {p_score:.1f}/10" if isinstance(p_score, float) else f"Pylint: {p_score}")

            error_summary_text = ""
            if parent.errors:
                first_error_str = str(parent.errors[0])
                if "failed" in first_error_str.lower() and (
                        "test" in first_error_str.lower() or "i/o" in first_error_str.lower()):
                    error_summary_text = " (failed I/O)"
                elif "syntax" in first_error_str.lower():
                    error_summary_text = " (syntax err)"
                else:
                    error_summary_text = " (had errors)"

            outcome_str = ', '.join(outcome_parts) + error_summary_text
            if not outcome_str.strip():  # Ensure there's some text if no specific scores/errors
                outcome_str = "Prior version."

            summary_entry = {
                "generation": parent.generation,
                "creation_method": parent.creation_method if parent.creation_method != "unknown" else "prev_gen",
                "outcome_summary": outcome_str
            }
            history_summary.append(summary_entry)
            current_prog_in_history = parent  # Move to the parent for the next iteration

        logger.debug(f"Returning {len(history_summary)} ancestral summaries for {program.id}.")
        return list(reversed(history_summary))  # Oldest relevant ancestor first for readability in prompt

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()