import logging
import asyncio
import random
import uuid  # Make sure uuid is imported, it's used for program_id
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from config import settings

# Import concrete agent implementations
from prompt_designer.prompt_designer_agent import PromptDesignerAgent
from code_generator.code_generator_agent import CodeGeneratorAgent
from evaluator_agent.evaluator_agent import EvaluatorAgent
from database_agent.database_agent import InMemoryDatabaseAgent  # Using InMemory for now
from database_agent.sqlite_database_agent import SQLiteDatabaseAgent # New!
from selection_controller.selection_controller_agent import SelectionControllerAgent

logger = logging.getLogger(__name__)


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

        self.crossover_rate = settings.CROSSOVER_RATE # e.g., 0.2 from settings
        self.min_parents_for_crossover = 2 # Typically 2

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        self.num_parents_to_select = self.population_size // 2

    async def initialize_population(self) -> List[Program]:  # Method_v2
        logger.info(
            f"Initializing population for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        initial_population = []

        # --- Start of new logic for seed code (v2) ---
        programs_to_generate_from_prompt = self.population_size
        start_index_for_prompt_generation = 0

        if self.task_definition.initial_seed_code and self.task_definition.improvement_mode == "general_refinement":
            logger.info(f"Using provided initial_seed_code for the first program.")
            program_id = f"{self.task_definition.id}_gen0_seed_prog0"
            seed_program = Program(
                id=program_id,
                code=self.task_definition.initial_seed_code,
                generation=0,
                status="unevaluated"  # Will be evaluated with the rest
            )
            initial_population.append(seed_program)
            await self.database.save_program(seed_program)

            programs_to_generate_from_prompt -= 1  # We've created one program from seed
            start_index_for_prompt_generation = 1  # Next program ID will start from 1

            # Optional: For now, we'll just generate the rest from scratch.
            # Future idea: Could prompt LLM to create variations of the seed_code here.
            # For example:
            # if programs_to_generate_from_prompt > 0:
            #     variation_prompt = f"Here is a piece of Python code:\n```python\n{self.task_definition.initial_seed_code}\n```\n" \
            #                        f"Generate {programs_to_generate_from_prompt} slight variations or alternative ways to structure this same logic, " \
            #                        f"keeping the function signature '{self.task_definition.function_name_to_evolve}' if present. " \
            #                        "Provide each variation as a complete Python code block, without explanations or markdown fences."
            #     # Then loop programs_to_generate_from_prompt times, calling self.code_generator.generate_code(variation_prompt)
            #     # This is more advanced and would require the LLM to give multiple distinct code blocks in one response, or multiple calls.
            #     # For simplicity in this step, we'll stick to generating the rest from the standard initial prompt.
            logger.info(f"Will generate {programs_to_generate_from_prompt} additional programs using initial prompt.")

        # --- End of new logic for seed code (v2) ---

        # Generate the remaining programs (or all, if no seed code) using prompts
        if programs_to_generate_from_prompt > 0:
            # Determine the prompt to use for initial generation
            # If it's 'general_refinement' but NO seed was given, this might be an ambiguous state.
            # For now, we assume if 'general_refinement' is chosen, 'initial_seed_code' *should* be provided.
            # If we are in 'task_focused' mode OR we are just filling population after a seed, use standard initial prompt.
            if self.task_definition.improvement_mode == "general_refinement" and not self.task_definition.initial_seed_code:
                logger.warning(
                    f"Task mode is 'general_refinement' but no initial_seed_code was provided. Proceeding with standard initial prompt for task '{self.task_definition.id}'. This might not be the intended behavior.")

            initial_prompt_text = self.prompt_designer.design_initial_prompt(self.task_definition) # <--- ADDED THIS ARGUMENT
            if not initial_prompt_text.strip() and self.task_definition.improvement_mode == "general_refinement":
                # This case should ideally be handled by PromptDesignerAgent adapting its initial prompt for refinement if needed.
                # For now, if PromptDesigner gives empty for refinement, we fall back or warn.
                logger.warning(
                    "Initial prompt from PromptDesigner was empty for general_refinement mode. This is unexpected.")
                # A very basic fallback if design_initial_prompt isn't aware of refinement mode yet:
                initial_prompt_text = (
                    f"You are an expert Python programmer. The primary goal is to work with and refine Python code. "
                    f"The current task is '{self.task_definition.description}'. "
                    f"Function name of interest is '{self.task_definition.function_name_to_evolve}'. "
                    f"Allowed imports: {self.task_definition.allowed_imports}. "
                    f"Provide a complete Python solution. No markdown fences or explanations."
                )

            for i in range(start_index_for_prompt_generation, self.population_size):
                # If we already added a seed program, and start_index is 1, this loop effectively runs fewer times.
                # Corrected loop condition:
                if len(initial_population) >= self.population_size:
                    break

                program_id = f"{self.task_definition.id}_gen0_prog{i}"
                logger.debug(
                    f"Generating initial program {len(initial_population) + 1}/{self.population_size} (ID: {program_id}) using prompt.")

                generated_code = await self.code_generator.generate_code(initial_prompt_text, temperature=0.8)

                program = Program(
                    id=program_id,
                    code=generated_code,
                    generation=0,
                    status="unevaluated",
                    task_id=self.task_definition.id  # <--- SET THE TASK ID!
                )
                initial_population.append(program)
                await self.database.save_program(program)

        logger.info(f"Initialized population with {len(initial_population)} programs.")
        return initial_population

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs = []
        evaluation_tasks = [self.evaluator.evaluate_program(prog, self.task_definition) for prog in population if prog.status != "evaluated"]
        
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            original_program = population[i] # Assumes order is maintained
            if isinstance(result, Exception):
                logger.error(f"Error evaluating program {original_program.id}: {result}", exc_info=result)
                original_program.status = "failed_evaluation"
                original_program.errors.append(str(result))
                evaluated_programs.append(original_program)
            else:
                evaluated_programs.append(result) # result is the evaluated Program object
            await self.database.save_program(evaluated_programs[-1]) # Update DB with evaluation results
            
        logger.info(f"Finished evaluating population. {len(evaluated_programs)} programs processed.")
        return evaluated_programs

    async def manage_evolutionary_cycle(self):  # Method_v3 (with Crossover)
        logger.info(
            f"Starting evolutionary cycle for task: {self.task_definition.id} (Mode: {self.task_definition.improvement_mode})")
        if isinstance(self.database, SQLiteDatabaseAgent):
            await self.database.setup_db()

        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            parents = self.selection_controller.select_parents(
                current_population,
                self.num_parents_to_select,
                # This might need to be self.population_size if all offspring come from selection
                self.task_definition
            )
            if not parents or len(parents) < 1:  # Need at least one parent for mutation
                logger.warning(
                    f"Generation {gen}: Not enough parents selected ({len(parents)}). Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents for reproduction.")

            offspring_population: List[Program] = []  # Explicitly typed

            # We aim to fill up to self.population_size with new offspring
            # Elitism is handled by select_survivors later by combining current_population and offspring_population
            num_offspring_to_generate = self.population_size

            generation_tasks = []

            for i in range(num_offspring_to_generate):
                # Decide between Crossover and Mutation
                if len(parents) >= self.min_parents_for_crossover and random.random() < self.crossover_rate:
                    # Perform Crossover
                    p1, p2 = random.sample(parents, 2)  # Select two distinct parents
                    child_id = f"{self.task_definition.id}_gen{gen}_cross{i}"
                    logger.debug(f"Attempting crossover for offspring {child_id} between {p1.id} and {p2.id}")
                    generation_tasks.append(self.generate_crossover_offspring(p1, p2, gen, child_id))
                else:
                    # Perform Mutation (or Bug Fix)
                    parent_for_mutation = random.choice(parents)
                    child_id = f"{self.task_definition.id}_gen{gen}_mut{i}"
                    logger.debug(
                        f"Attempting mutation/bug_fix for offspring {child_id} from parent {parent_for_mutation.id}")
                    # generate_offspring handles mutation/bug_fix logic
                    generation_tasks.append(self.generate_offspring(parent_for_mutation, gen, child_id))

            if generation_tasks:
                generated_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
                for result in generated_results:
                    if isinstance(result, Program):
                        offspring_population.append(result)
                        await self.database.save_program(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Error during offspring generation task: {result}", exc_info=result)
                    # If result is None (generation method decided not to produce offspring), it's skipped.

            logger.info(f"Generation {gen}: Generated {len(offspring_population)} new offspring candidates.")

            if offspring_population:
                offspring_population = await self.evaluate_population(offspring_population)

            # Combine current population (which might contain elites from previous gen) with new offspring
            # The selection_controller.select_survivors will then pick the best to form the next generation.
            # Note: current_population here is from *before* new offspring were made.
            # If elitism is handled by select_parents carrying them over, that's one way.
            # More commonly, select_survivors takes (previous_generation_survivors + new_offspring)
            # Let's assume current_population is the survivors from the *end* of generation gen-1.

            current_population = self.selection_controller.select_survivors(
                current_population,  # Survivors from gen-1
                offspring_population,  # Newly created and evaluated children for gen
                self.population_size,
                self.task_definition
            )
            logger.info(f"Generation {gen}: New population size after survival: {len(current_population)}.")

            current_population = self.selection_controller.select_survivors(
                current_population,  # This is the population from the *previous* generation (could include elites)
                offspring_population,  # These are the newly generated and evaluated offspring
                self.population_size,
                self.task_definition  # <--- ADDED THIS ARGUMENT
            )
            logger.info(f"Generation {gen}: New population size after survival: {len(current_population)}.")

            # Logging best program of the generation
            if current_population:
                # Sort current_population by the same key used in selection to find the 'best'
                # This requires SelectionControllerAgent to expose its sorting key or for TaskManager to replicate it.
                # For now, let's use a simplified sort here for logging, assuming higher correctness is better.
                # A more robust way would be to have selection_controller return the best, or use the same sort key.
                temp_sorted_for_log = sorted(
                    current_population,
                    key=lambda p: (
                        p.fitness_scores.get("correctness", 0.0),
                        # Add other primary metrics if needed for a more accurate "best" log here
                        -p.fitness_scores.get("runtime_ms", float('inf'))  # Example tie-breaker
                    ),
                    reverse=True
                )
                best_program_this_gen = temp_sorted_for_log[0]
                logger.info(
                    f"Generation {gen}: Best program in new pop: ID={best_program_this_gen.id}, Fitness={best_program_this_gen.fitness_scores}")
            else:
                logger.warning(
                    f"Generation {gen}: No programs in current population after survival selection. Ending evolution.")
                break  # End evolution if population becomes empty

        logger.info("Evolutionary cycle completed.")
        # --- Fetching overall best program using new selection logic from database ---
        # The DatabaseAgent's get_best_programs might need an update to use a similar dynamic sorting key
        # or we fetch all and sort here. For now, let's assume it has a reasonable default or we enhance it later.
        # A simple way for now: fetch more programs and re-sort them using SelectionController's logic.
        all_evaluated_programs = await self.database.get_all_programs()  # Assuming get_all_programs is now in interface
        if all_evaluated_programs:
            # Filter for only evaluated programs *before* sorting
            candidate_programs = [p for p in all_evaluated_programs if p.status == "evaluated"]
            if candidate_programs:
                # Use the new public sorting method from the selection controller
                final_best_sorted = self.selection_controller.sort_programs(candidate_programs,
                                                                            self.task_definition)  # Changed _v5
                if final_best_sorted:
                    best_overall_program = final_best_sorted[0]
                    logger.info(
                        f"Overall Best Program Found: ID={best_overall_program.id}, Code:\n{best_overall_program.code}\nFitness: {best_overall_program.fitness_scores}, Generation: {best_overall_program.generation}")
                    return [best_overall_program]
                else:
                    logger.info("No evaluated programs left after filtering to sort.")
            else:
                logger.info("No evaluated programs found to determine an overall best.")
        else:
            logger.info("No programs found in the database at the end of evolution.")
        return []

    async def generate_crossover_offspring(self, parent1: Program, parent2: Program, generation_num: int,
                                           child_id: str) -> Optional[Program]:  # v1.0.0
        logger.debug(
            f"Generating crossover offspring {child_id} from parents {parent1.id} & {parent2.id} for gen {generation_num}")

        crossover_prompt = self.prompt_designer.design_crossover_prompt(self.task_definition, parent1, parent2)

        # For crossover, we expect a full new code block
        generated_code = await self.code_generator.generate_code(  # Use generate_code directly
            prompt=crossover_prompt,
            temperature=0.7,  # May want a slightly different temperature for creative synthesis
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
        Program]:  # Method_v2.1 (Completed)
        logger.debug(
            f"Generating mutation/bug_fix offspring {child_id} from parent {parent.id} for gen {generation_num}")

        mutation_prompt_str: str = ""  # Initialize to ensure it's always a string
        prompt_type: str = "mutation"  # Default creation method

        # Prepare evaluation feedback for the prompt designer, ensuring all relevant keys are present if possible
        parent_feedback_cleaned = {
            "correctness": parent.fitness_scores.get("correctness"),
            "runtime_ms": parent.fitness_scores.get("runtime_ms"),
            "pylint_score": parent.fitness_scores.get("pylint_score"),
            "cyclomatic_complexity_avg": parent.fitness_scores.get("cyclomatic_complexity_avg"),
            "maintainability_index": parent.fitness_scores.get("maintainability_index"),
            "passed_tests": parent.fitness_scores.get("passed_tests"),
            "total_tests": parent.fitness_scores.get("total_tests"),
            "errors": parent.errors,  # This is a list
        }
        # Clean out None values to keep the prompt tidy, but ensure 'errors' list is always there.
        parent_feedback_for_prompt = {k: v for k, v in parent_feedback_cleaned.items() if
                                      v is not None or k == "errors"}

        # Decide if this should be a bug-fix attempt or a general mutation
        # A common heuristic: if correctness is very low and there are errors, prioritize bug fixing.
        # The threshold (e.g., 0.1 for correctness) can be a setting later if needed.
        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < 0.1:
            primary_error = parent.errors[0]  # Take the first error as the primary one for the prompt
            # We don't have detailed execution_output readily available here unless EvaluatorAgent populates it
            # into parent.errors in a structured way or another field. For now, pass None.
            execution_details_for_prompt = None

            logger.info(f"Designing bug-fix prompt for parent {parent.id}. Error: {primary_error}")
            mutation_prompt_str = self.prompt_designer.design_bug_fix_prompt(
                task=self.task_definition,  # Pass the task definition
                program=parent,
                error_message=primary_error,
                execution_output=execution_details_for_prompt  # This is often None
            )
            prompt_type = "bug_fix"
        else:
            logger.info(
                f"Designing mutation prompt for parent {parent.id} (Mode: {self.task_definition.improvement_mode}).")
            mutation_prompt_str = self.prompt_designer.design_mutation_prompt(
                task=self.task_definition,  # Correct: This is the first parameter after self
                parent_program=parent,  # <-- CORRECTED! Was 'program=parent'
                evaluation_feedback=parent_feedback_for_prompt
            )
            prompt_type = "mutation"

        # Generate code (or rather, a diff) using the designed prompt
        # The CodeGeneratorAgent's execute method with output_format="diff" will attempt to apply the diff.
        # The result, `final_code_after_diff`, will be the complete modified code if successful.
        final_code_after_diff = await self.code_generator.execute(
            prompt=mutation_prompt_str,
            temperature=0.75,  # Could be a setting
            output_format="diff",
            parent_code_for_diff=parent.code
        )

        # Check if the generation resulted in no change, empty code, or if the LLM returned the raw diff (error)
        if not final_code_after_diff.strip():
            logger.warning(
                f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in EMPTY code. Skipping.")
            return None
        if final_code_after_diff == parent.code:
            logger.warning(
                f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in NO CHANGE to the code. Skipping.")
            return None

        # A simple heuristic: if the CodeGeneratorAgent failed to apply the diff and returned the raw diff text
        if "<<<<<<< SEARCH" in final_code_after_diff and ">>>>>>> REPLACE" in final_code_after_diff:
            logger.warning(
                f"Offspring generation for {parent.id} ({prompt_type}) returned raw diff text, indicating diff application likely failed or LLM didn't follow instructions. Diff text: {final_code_after_diff[:300]}... Skipping.")
            return None

        offspring = Program(
            id=child_id,
            code=final_code_after_diff,  # This is the modified code after diff application
            generation=generation_num,
            parent_id=parent.id,
            parent_ids=[parent.id] if parent.id else [],  # For consistency with crossover, if we merge parent tracking
            status="unevaluated",
            task_id=self.task_definition.id,
            creation_method=prompt_type
        )
        logger.info(f"Successfully generated {prompt_type} offspring {offspring.id} from parent {parent.id}.")
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

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()