import logging
import asyncio
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
from selection_controller.selection_controller_agent import SelectionControllerAgent

logger = logging.getLogger(__name__)


class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()

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
                    status="unevaluated"
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

    async def manage_evolutionary_cycle(self):  # Method_v2 (minor argument changes)
        logger.info(
            f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}... (Mode: {self.task_definition.improvement_mode})")  # Added mode log
        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            # 1. Selection
            # OLD: parents = self.selection_controller.select_parents(current_population, self.num_parents_to_select)
            # NEW (v2): Pass self.task_definition
            parents = self.selection_controller.select_parents(
                current_population,
                self.num_parents_to_select,
                self.task_definition  # <--- ADDED THIS ARGUMENT
            )
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")

            # 2. Crossover (simplified: not implemented, LLM mutation is primary)
            # 3. Mutation (generating offspring)
            offspring_population = []
            # Calculate num_offspring_per_parent more carefully to aim for population_size
            if not parents:  # Should be caught by above, but as a safeguard
                num_offspring_to_generate = self.population_size
            else:
                num_offspring_to_generate = self.population_size - len(
                    parents)  # How many new individuals we want if elites are kept
                if num_offspring_to_generate <= 0:  # This can happen if elitism_count is high
                    num_offspring_to_generate = self.population_size  # Fallback to try and generate full pop if elites filled it.

            generation_tasks = []
            # Ensure we have parents to generate from for the loop logic
            if parents:
                # Distribute offspring generation among parents
                # This logic tries to ensure each parent contributes, up to the number of offspring we need.
                # It's a bit simplified; more complex schemes exist.
                parent_idx = 0
                for i in range(num_offspring_to_generate):  # Generate needed number of offspring
                    parent_for_this_child = parents[parent_idx % len(parents)]
                    child_id = f"{self.task_definition.id}_gen{gen}_child{i}"  # Use i for unique child ID in this gen
                    generation_tasks.append(self.generate_offspring(parent_for_this_child, gen, child_id))
                    parent_idx += 1
            elif num_offspring_to_generate > 0:  # No parents, but we need to fill population (e.g. if elitism failed)
                # This case is less ideal. We'd be generating from scratch again if no parents.
                # For now, the earlier 'break' if no parents is the primary path. This is a fallback.
                logger.warning(
                    f"Generation {gen}: No parents, but attempting to generate {num_offspring_to_generate} new individuals from scratch.")
                for i in range(num_offspring_to_generate):
                    child_id = f"{self.task_definition.id}_gen{gen}_child_from_scratch_{i}"
                    # This would ideally use a different prompt mechanism if we're starting from scratch mid-evolution
                    # For now, let's assume generate_offspring can handle a None parent or we rely on earlier check.
                    # The current generate_offspring expects a parent. So this path needs more thought if we hit it.
                    # The code currently breaks if parents is empty, so this path won't be hit with current logic.
                    pass  # Placeholder for if we wanted to handle "generate new from scratch"

            if generation_tasks:  # Only proceed if there are tasks to generate offspring
                generated_offspring_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

                # Create a new list for successfully generated offspring for this generation
                newly_generated_offspring: List[Program] = []  # Explicitly typed _v4

                for result in generated_offspring_results:
                    if isinstance(result, Exception):
                        # It's an exception object, log it as an error
                        logger.error(f"Error during offspring generation task: {result}", exc_info=result)
                    elif isinstance(result, Program):  # Check if it's a Program object _v4
                        # It's a successfully created Program object
                        newly_generated_offspring.append(result)
                        await self.database.save_program(result)  # Save the valid program
                    # If result is None (because generate_offspring chose not to create one), we just ignore it.
                    # generate_offspring returns Optional[Program]

                # Now, offspring_population should only contain valid Program objects from this gather operation
                offspring_population.extend(newly_generated_offspring)  # Add to the main list for the generation _v4
                # Note: offspring_population was initialized as [] earlier in the generation loop.

            logger.info(
                f"Generation {gen}: Generated {len(offspring_population)} valid offspring this round.")  # Updated log message

            # If no new offspring were generated AND no parents were carried over (e.g. if elitism is 0 and all parents failed to produce offspring)
            # This situation needs careful handling to avoid empty populations.
            # The current survivor selection will handle combining current_pop (which might have elites) and offspring.

            # 4. Evaluation of Offspring
            if offspring_population:  # Only evaluate if there are offspring
                # Now offspring_population is guaranteed to be List[Program] (or empty)
                offspring_population = await self.evaluate_population(offspring_population)
            else:
                logger.info(f"Generation {gen}: No new valid offspring to evaluate this round.")

            # 5. Survivor Selection
            # OLD: current_population = self.selection_controller.select_survivors(current_population, offspring_population, self.population_size)
            # NEW (v2): Pass self.task_definition
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
    
    async def generate_offspring(self, parent: Program, generation_num: int, child_id:str) -> Optional[Program]:
        logger.debug(f"Generating offspring from parent {parent.id} for generation {generation_num}")

        prompt_type = "mutation"
        # Use self.task_definition which is available in the class instance
        current_task_def = self.task_definition

        # Prepare evaluation feedback for the prompt designer
        # The evaluation_feedback dict should match what _format_evaluation_feedback expects
        parent_feedback = {
            "correctness": parent.fitness_scores.get("correctness"),
            "runtime_ms": parent.fitness_scores.get("runtime_ms"),
            # We need to add our new scores here!
            "pylint_score": parent.fitness_scores.get("pylint_score"),
            "cyclomatic_complexity_avg": parent.fitness_scores.get("cyclomatic_complexity_avg"),
            "maintainability_index": parent.fitness_scores.get("maintainability_index"),
            "passed_tests": parent.fitness_scores.get("passed_tests"),  # Added for the new feedback format
            "total_tests": parent.fitness_scores.get("total_tests"),  # Added for the new feedback format
            "errors": parent.errors,
        }
        # Remove None values from feedback to keep prompt cleaner
        parent_feedback_cleaned = {k: v for k, v in parent_feedback.items() if v is not None}

        mutation_prompt = ""
        # Use self.task_definition which is available in the class instance
        current_task_def = self.task_definition  # This is good

        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < 0.1:  # If significant errors
            primary_error = parent.errors[0]
            execution_details = None  # Placeholder, extract if available

            # MODIFIED CALL _v6 for this file
            mutation_prompt = self.prompt_designer.design_bug_fix_prompt(
                current_task_def,  # Pass the TaskDefinition object
                program=parent,
                error_message=primary_error,
                execution_output=execution_details
            )
            logger.info(f"Attempting bug fix for parent {parent.id} using diff. Error: {primary_error}")
            prompt_type = "bug_fix"
        else:
            # Pass current_task_def to design_mutation_prompt
            mutation_prompt = self.prompt_designer.design_mutation_prompt(
                current_task_def,  # Pass the task definition
                parent,
                evaluation_feedback=parent_feedback_cleaned
            )
            logger.info(
                f"Attempting mutation for parent {parent.id} (Mode: {current_task_def.improvement_mode}) using diff.")

        generated_code = await self.code_generator.execute(
            prompt=mutation_prompt,
            temperature=0.75,
            output_format="diff",
            parent_code_for_diff=parent.code
        )

        if not generated_code.strip() or generated_code == parent.code:
            logger.warning(
                f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in no change or empty code. Skipping.")
            return None

        if "<<<<<<< SEARCH" in generated_code and "=======" in generated_code and ">>>>>>> REPLACE" in generated_code:
            logger.warning(
                f"Offspring generation for parent {parent.id} ({prompt_type}) seems to have returned raw diff. Skipping. Content:\n{generated_code[:500]}")
            return None

        if "# Error:" in generated_code[:100]:
            logger.warning(
                f"Failed to generate valid code for offspring of {parent.id} ({prompt_type}). LLM Output indicates error: {generated_code[:200]}")
            return None

        offspring = Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_id=parent.id,
            status="unevaluated"
        )
        logger.info(f"Successfully generated offspring {offspring.id} from parent {parent.id} ({prompt_type}).")
        return offspring

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()

# Example Usage (for testing this agent directly):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # To see DEBUG logs from specific modules, you can do:
    # logging.getLogger("alpha_evolve_pro.code_generator.agent").setLevel(logging.DEBUG)
    # logging.getLogger("alpha_evolve_pro.prompt_designer.agent").setLevel(logging.DEBUG)

    task_manager = TaskManagerAgent(task_definition=sample_task) # Pass sample_task here

    # Define a simple task
    sample_task = TaskDefinition(
        id="sum_list_task_001",
        description="Write a Python function called `solve(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [-1, 0, 1], "output": 0},
            {"input": [10, 20, 30, 40, 50], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness", "goal": "maximize"},
        initial_code_prompt = "Please provide a Python function `solve(numbers)` that sums a list of integers. Handle empty lists by returning 0."
    )
    
    # Reduce generations/population for quicker test
    task_manager.num_generations = 3 # settings.GENERATIONS = 3
    task_manager.population_size = 5 # settings.POPULATION_SIZE = 5
    task_manager.num_parents_to_select = 2 # settings.POPULATION_SIZE // 2 

    async def run_task():
        # Ensure GEMINI_API_KEY is in your .env file or environment
        try:
            best_programs = await task_manager.manage_evolutionary_cycle() # Removed sample_task argument
            if best_programs:
                print(f"\n*** Evolution Complete! Best program found: ***")
                print(f"ID: {best_programs[0].id}")
                print(f"Generation: {best_programs[0].generation}")
                print(f"Fitness: {best_programs[0].fitness_scores}")
                print(f"Code:\n{best_programs[0].code}")
            else:
                print("\n*** Evolution Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the task management cycle.", exc_info=True)

    asyncio.run(run_task()) 