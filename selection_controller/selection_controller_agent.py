import random
import logging
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# We need TaskDefinition if we're going to make decisions based on it.
from core.interfaces import SelectionControllerInterface, Program, BaseAgent, TaskDefinition # Added TaskDefinition_v2
from config import settings

logger = logging.getLogger(__name__)


class SelectionControllerAgent(SelectionControllerInterface, BaseAgent):
    def __init__(self):
        super().__init__()  # Call BaseAgent's __init__
        self.elitism_count = settings.ELITISM_COUNT
        logger.info(f"SelectionControllerAgent initialized with elitism_count: {self.elitism_count}")

    # Method_v3 (Updated for Ruff, ensuring lower ruff_violations is better)
    def _get_program_sort_key(self, program: Program, task: TaskDefinition) -> Tuple:
        """
        Creates a sort key for a program. Higher tuple values mean "better" for sorting (reverse=True).
        Primary: Correctness (higher is better)
        Secondary: Ruff Violations (lower is better, so we use -violations)
        Tertiary: Primary Focus Metrics from TaskDefinition (handled based on their optimization direction)
        Default Tie-breakers: Runtime (lower is better, so -runtime), Generation (higher is better)
        """
        key_parts = []

        # 1. Correctness (higher is better)
        correctness = program.fitness_scores.get("correctness", settings.DEFAULT_METRIC_VALUE.get("correctness", 0.0))
        key_parts.append(correctness)

        # 2. Ruff Violations (lower is better)
        # To sort descending by "goodness" (reverse=True later), and lower ruff_violations is better,
        # we use -ruff_violations. E.g., 0 violations -> key part 0; 5 violations -> key part -5.
        # When sorted descending, 0 comes before -5.
        ruff_violations = program.fitness_scores.get("ruff_violations",
                                                     settings.DEFAULT_METRIC_VALUE.get("ruff_violations", float('inf')))
        key_parts.append(-ruff_violations)

        # 3. Primary Focus Metrics from TaskDefinition
        if task.primary_focus_metrics:
            for metric_name in task.primary_focus_metrics:
                # Skip metrics already explicitly handled above to avoid double-counting/weighting
                if metric_name in ["correctness", "ruff_violations"]:
                    continue

                value = program.fitness_scores.get(metric_name, settings.DEFAULT_METRIC_VALUE.get(metric_name, 0.0))
                # METRIC_OPTIMIZATION_DIRECTION: True if higher is better, False if lower is better
                higher_is_better = settings.METRIC_OPTIMIZATION_DIRECTION.get(metric_name, True)

                if higher_is_better:
                    key_parts.append(value)  # Add directly, reverse=True will handle descending sort
                else:  # Lower is better
                    key_parts.append(
                        -value)  # Negate so that smaller actual values result in larger key components for descending sort

        # 4. Default tie-breakers (if not already primary focus metrics)
        # Runtime (lower is better)
        if "runtime_ms" not in (task.primary_focus_metrics or []):  # Avoid double-counting
            runtime = program.fitness_scores.get("runtime_ms",
                                                 settings.DEFAULT_METRIC_VALUE.get("runtime_ms", float('inf')))
            key_parts.append(-runtime)  # Negate for descending sort of "goodness"

        # Generation (higher is better - favors newer solutions in ties)
        # This helps ensure that if all else is equal, newer programs might be preferred slightly.
        key_parts.append(program.generation)

        # Debug log for clarity during development
        # logger.debug(
        #     f"ProgID: {program.id}, Gen: {program.generation}, "
        #     f"Correctness: {correctness:.2f}, RuffV: {ruff_violations}, "
        #     f"Runtime: {program.fitness_scores.get('runtime_ms', 'N/A')}, "
        #     f"Sort Key: {key_parts}"
        # )
        return tuple(key_parts)

    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[
        Program]:  # Method_v3.1 (Uses updated _get_program_sort_key)
        if not programs:
            return []
        # Sorts programs. reverse=True means programs with "better" (larger) key tuples come first.
        return sorted(
            programs,
            key=lambda p: self._get_program_sort_key(p, task),
            reverse=True
        )

    def select_parents(self, population: List[Program], num_parents: int, task: TaskDefinition) -> List[
        Program]:  # Method_v2.1 (Uses updated sort_programs)
        logger.info(
            f"Starting parent selection. Pop size: {len(population)}, Num parents to select: {num_parents}, Task Mode: {task.improvement_mode}")

        if not population:
            logger.warning("Parent selection called with empty population. Returning empty list.")
            return []
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []

        # Sort population by fitness using the centralized sort_programs method
        sorted_population = self.sort_programs(population, task)

        # Log top few for debugging, if needed
        # top_program_ids_and_keys = [(p.id, self._get_program_sort_key(p, task)) for p in sorted_population[:min(5, len(sorted_population))]]
        # logger.debug(f"Population sorted for parent selection. Top candidates (IDs, SortKey): {top_program_ids_and_keys}")

        parents: List[Program] = []

        # 1. Elitism: Select the top N unique individuals based on the sort order
        # Elitism_count should not exceed num_parents or available population
        actual_elitism_count = min(self.elitism_count, num_parents, len(sorted_population))

        elite_candidates = sorted_population[:actual_elitism_count]
        parents.extend(elite_candidates)
        logger.info(f"Selected {len(elite_candidates)} elite parents: {[p.id for p in elite_candidates]}")

        remaining_slots = num_parents - len(parents)
        if remaining_slots <= 0:
            logger.info("Elitism filled all parent slots or no more parents needed.")
            return parents  # Return only elites if they fill all slots

        # Candidates for further selection (non-elite part of sorted_population)
        # Make sure not to re-select elites if roulette_candidates come from the full sorted_population
        roulette_candidate_pool = [p for p in sorted_population if p not in parents]

        if not roulette_candidate_pool:
            logger.warning("No candidates left for further selection after elitism. Returning current elite parents.")
            return parents

        # Fitness-Proportionate Selection (Roulette Wheel) for remaining slots
        # Using 'correctness' for roulette simplicity, as multi-objective is complex for roulette.
        # Add a small epsilon to fitness to handle zero-fitness individuals if all are zero.
        fitness_values_roulette = [p.fitness_scores.get("correctness", 0.0) + 0.0001 for p in roulette_candidate_pool]
        total_fitness_roulette = sum(fitness_values_roulette)

        logger.debug(
            f"Total 'correctness' fitness for roulette wheel (among {len(roulette_candidate_pool)} candidates): {total_fitness_roulette:.4f}")

        if total_fitness_roulette <= (0.0001 * len(roulette_candidate_pool)) + 1e-9:  # Handles all effectively zero
            logger.warning(
                "All roulette candidates have effectively zero correctness. Selecting randomly from them for remaining slots.")
            num_to_select_randomly = min(remaining_slots, len(roulette_candidate_pool))
            if num_to_select_randomly > 0:
                random_parents_from_pool = random.sample(roulette_candidate_pool, num_to_select_randomly)
                parents.extend(random_parents_from_pool)
                logger.info(f"Selected {len(random_parents_from_pool)} parents randomly due to zero total correctness.")
        else:
            for _ in range(remaining_slots):
                if not roulette_candidate_pool: break  # No more candidates

                pick = random.uniform(0, total_fitness_roulette)
                current_sum_for_pick = 0
                chosen_parent_for_slot = None

                # Iterate through candidates and their fitness for roulette pick
                for idx, program in enumerate(roulette_candidate_pool):
                    current_sum_for_pick += fitness_values_roulette[idx]
                    if current_sum_for_pick >= pick:
                        chosen_parent_for_slot = program
                        break

                if chosen_parent_for_slot:
                    parents.append(chosen_parent_for_slot)
                    # Optional: if sampling without replacement from roulette_candidate_pool, remove it here
                    # and adjust total_fitness_roulette and fitness_values_roulette.
                    # For simplicity, this example does sampling with replacement from the pool for each slot.
                else:  # Fallback (should be rare if total_fitness > 0)
                    if roulette_candidate_pool:  # Ensure pool is not empty
                        logger.debug("Roulette pick fallback: choosing random from pool.")
                        parents.append(random.choice(roulette_candidate_pool))

        # Ensure we don't exceed num_parents due to any logic edge cases (though above should handle it)
        final_parents = parents[:num_parents]
        logger.info(f"Total parents selected: {len(final_parents)}. IDs: {[p.id for p in final_parents]}")
        return final_parents

    def select_survivors(self, current_population: List[Program], offspring_population: List[Program],
                         population_size: int, task: TaskDefinition) -> List[
        Program]:  # Method_v2.1 (Uses updated sort_programs)
        logger.info(
            f"Starting survivor selection. Current pop: {len(current_population)}, Offspring pop: {len(offspring_population)}, Target pop_size: {population_size}")

        combined_population = current_population + offspring_population
        logger.debug(f"Combined population size for survivor selection: {len(combined_population)}")

        if not combined_population:
            logger.warning("Survivor selection called with empty combined population. Returning empty list.")
            return []

        # Sort the combined population by fitness (higher key value is better)
        sorted_combined_population = self.sort_programs(combined_population, task)

        # Select the top 'population_size' individuals
        # Using a set to ensure uniqueness of program IDs if there's any chance of duplicates
        # (though program IDs should generally be unique from generation process).
        survivors: List[Program] = []
        seen_ids_for_survivors = set()
        for program in sorted_combined_population:
            if len(survivors) < population_size:
                if program.id not in seen_ids_for_survivors:
                    survivors.append(program)
                    seen_ids_for_survivors.add(program.id)
            else:
                break  # Reached desired population size

        logger.info(f"Selected {len(survivors)} survivors. IDs: {[p.id for p in survivors]}")
        return survivors

    async def execute(self, action: str, **kwargs) -> Any:  # Unchanged from your previous structure
        task = kwargs.get('task')
        if not isinstance(task, TaskDefinition):  # Make sure task is always provided for selection methods
            raise ValueError("A TaskDefinition object must be provided in kwargs as 'task' for selection operations.")

        if action == "select_parents":
            population = kwargs.get('population')
            num_parents = kwargs.get('num_parents')
            if population is None or num_parents is None:
                raise ValueError("Missing 'population' or 'num_parents' for 'select_parents' action.")
            return self.select_parents(population, num_parents, task)
        elif action == "select_survivors":
            current_population = kwargs.get('current_population')
            offspring_population = kwargs.get('offspring_population')
            population_size = kwargs.get('population_size')
            if current_population is None or offspring_population is None or population_size is None:
                raise ValueError("Missing arguments for 'select_survivors' action.")
            return self.select_survivors(current_population, offspring_population, population_size, task)
        elif action == "sort_programs":  # Added this action if TaskManager needs to sort externally
            programs_to_sort = kwargs.get('programs')
            if programs_to_sort is None:
                raise ValueError("Missing 'programs' for 'sort_programs' action.")
            return self.sort_programs(programs_to_sort, task)
        else:
            logger.error(f"Unknown action '{action}' for SelectionControllerAgent.execute().")
            raise ValueError(f"Unknown action: {action}")