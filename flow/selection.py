# selection_controller/selection.py
# Version: 1.1.1 (Corrected potential editor misinterpretation of list.append)

import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from core.interfaces import SelectionControllerInterface, Program, BaseAgent, TaskDefinition
from config import settings

logger = logging.getLogger(__name__)


class EvoSelector(SelectionControllerInterface, BaseAgent):
    def __init__(self):
        super().__init__()
        self.elitism_count = settings.ELITISM_COUNT
        logger.info(f"EvoSelector initialized with elitism_count: {self.elitism_count}")

    def _get_program_sort_key(self, program: Program, task: TaskDefinition) -> Tuple: # Method_v4.0.2
        """
        Creates a sort key for a program. Higher tuple values mean "better" for sorting (reverse=True).
        Priority:
        1. runs_without_error (True is infinitely better)
        2. ai_review_score (higher is better)  <--- UPDATED NAME
        3. correctness (I/O tests, higher is better)
        4. -ruff_violations (lower is better)
        5. Primary Focus Metrics from TaskDefinition
        6. Default Tie-breakers: -runtime_ms (lower is better), generation (higher is better)
        """
        key_parts: List[Union[float, int]] = []

        # --- START NEW DETAILED LOGGING ---
        logger.debug(
            f"[_get_program_sort_key] Evaluating Program ID: {program.id}, Gen: {program.generation}, Status: {program.status}"
        )
        logger.debug(f"[_get_program_sort_key] Raw fitness_scores for {program.id}: {program.fitness_scores}")

        runs_ok_value_from_fitness = program.fitness_scores.get("runs_without_error", False)  # Default to False
        logger.debug(
            f"[_get_program_sort_key] For Program ID: {program.id}, 'runs_without_error' from fitness_scores: {runs_ok_value_from_fitness} (type: {type(runs_ok_value_from_fitness)})"
        )
        # --- END NEW DETAILED LOGGING ---

        # 1. Did it run without error?
        runs_ok_score = 1.0 if runs_ok_value_from_fitness else 0.0  # Use the fetched value
        key_parts.append(runs_ok_score)

        # 2. AI Review Score (higher is better)  <--- UPDATED SECTION
        ai_review_score_value = program.fitness_scores.get(
            "ai_review_score", # Use the new key
            settings.DEFAULT_METRIC_VALUE.get("ai_review_score", 0.0) # Use the new key
        )
        key_parts.append(ai_review_score_value) # Use the new local variable name

        # 3. Correctness
        correctness = program.fitness_scores.get("correctness", settings.DEFAULT_METRIC_VALUE.get("correctness", 0.0))
        key_parts.append(correctness)

        # 4. Ruff Violations
        ruff_violations = program.fitness_scores.get("ruff_violations",
                                                     settings.DEFAULT_METRIC_VALUE.get("ruff_violations", float('inf')))
        key_parts.append(-ruff_violations if ruff_violations != float('inf') else -999999.0)

        # 5. Primary Focus Metrics from TaskDefinition
        if task.primary_focus_metrics:
            for metric_name in task.primary_focus_metrics:
                # Skip metrics already explicitly handled above
                if metric_name in ["runs_without_error", "ai_review_score", "correctness", "ruff_violations"]: # UPDATED KEY
                    continue

                value = program.fitness_scores.get(metric_name, settings.DEFAULT_METRIC_VALUE.get(metric_name, 0.0))
                higher_is_better = settings.METRIC_OPTIMIZATION_DIRECTION.get(metric_name, True)

                value = program.fitness_scores.get(metric_name, settings.DEFAULT_METRIC_VALUE.get(metric_name, 0.0))
                higher_is_better = settings.METRIC_OPTIMIZATION_DIRECTION.get(metric_name, True)
                if higher_is_better:
                    key_parts.append(value if isinstance(value, (int, float)) else 0.0)
                else:
                    if value == float('inf'):
                        key_parts.append(-999999.0)
                    elif isinstance(value, (int, float)):
                        key_parts.append(-value)
                    else:
                        key_parts.append(0.0)

        # 6. Default tie-breakers (as before)
        if "runtime_ms" not in (task.primary_focus_metrics or []):  # ...
            runtime = program.fitness_scores.get("runtime_ms",
                                                 settings.DEFAULT_METRIC_VALUE.get("runtime_ms", float('inf')))
            key_parts.append(-runtime if runtime != float('inf') else -999999.0)
        key_parts.append(program.generation)

        # Optional: Update the debug log message if it was uncommented and used
        # logger.debug(
        #     f"ProgID: {program.id}, Gen: {program.generation}, "
        #     f"RunsOK: {runs_ok_score}, AIReview: {ai_review_score_value:.1f}, Correct: {correctness:.2f}, RuffV: {ruff_violations}, " # UPDATED
        #     f"Sort Key: {key_parts}"
        # )
        return tuple(key_parts)

    # sort_programs, get_parents, get_survivors, and execute methods remain as in Version 1.1.0
    # (as they rely on _get_program_sort_key which is now corrected)

    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[Program]:
        if not programs:
            return []
        return sorted(
            programs,
            key=lambda p: self._get_program_sort_key(p, task),
            reverse=True
        )

    def get_parents(self, population: List[Program], num_parents: int, task: TaskDefinition) -> List[Program]:
        logger.info(
            f"Starting parent selection. Pop size: {len(population)}, Num parents to select: {num_parents}, Task Mode: {task.improvement_mode}"
        )
        if not population:
            logger.warning("Parent selection called with empty population. Returning empty list.")
            return []
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []

        sorted_population = self.sort_programs(population, task)

        top_program_details = []
        for p_idx, p_val in enumerate(sorted_population[:min(5, len(sorted_population))]):
            key_tuple = self._get_program_sort_key(p_val, task)
            formatted_key = tuple(f"{x:.2f}" if isinstance(x, float) else x for x in key_tuple)
            top_program_details.append(
                f"ID:{p_val.id} Key:{formatted_key} AIReviewScore:{p_val.fitness_scores.get('ai_review_score', 'N/A')}")
        logger.debug(f"Population sorted for parent selection. Top candidates: [{'; '.join(top_program_details)}]")

        parents: List[Program] = []
        actual_elitism_count = min(self.elitism_count, num_parents, len(sorted_population))
        elite_candidates = sorted_population[:actual_elitism_count]
        parents.extend(elite_candidates)
        logger.info(f"Selected {len(elite_candidates)} elite parents: {[p.id for p in elite_candidates]}")

        remaining_slots = num_parents - len(parents)
        if remaining_slots <= 0:
            logger.info("Elitism filled all parent slots or no more parents needed.")
            return parents

        roulette_candidate_pool = [p for p in sorted_population if p not in parents]
        if not roulette_candidate_pool:
            logger.warning("No candidates left for further selection after elitism. Returning current elite parents.")
            return parents

        fitness_values_for_roulette = []
        for p_roulette in roulette_candidate_pool:
            score = p_roulette.fitness_scores.get("ai_review_score")
            if score is None:
                score = p_roulette.fitness_scores.get("correctness", 0.0)
            fitness_values_for_roulette.append(max(score, 0.0) + 0.0001)

        total_fitness_roulette = sum(fitness_values_for_roulette)
        logger.debug(
            f"Total fitness for roulette wheel (among {len(roulette_candidate_pool)} candidates, using ai review score or correctness): {total_fitness_roulette:.4f}")

        if total_fitness_roulette <= (0.0001 * len(roulette_candidate_pool)) + 1e-9:
            logger.warning(
                "All roulette candidates have effectively zero fitness for roulette. Selecting randomly from them.")
            num_to_select_randomly = min(remaining_slots, len(roulette_candidate_pool))
            if num_to_select_randomly > 0:
                random_parents_from_pool = random.sample(roulette_candidate_pool, num_to_select_randomly)
                parents.extend(random_parents_from_pool)
                logger.info(f"Selected {len(random_parents_from_pool)} parents randomly due to zero total fitness.")
        else:
            for _ in range(remaining_slots):
                if not roulette_candidate_pool: break
                pick = random.uniform(0, total_fitness_roulette)
                current_sum_for_pick = 0
                chosen_parent_for_slot = None
                for idx, program_in_pool in enumerate(roulette_candidate_pool):
                    current_sum_for_pick += fitness_values_for_roulette[idx]
                    if current_sum_for_pick >= pick:
                        chosen_parent_for_slot = program_in_pool
                        break
                if chosen_parent_for_slot:
                    parents.append(chosen_parent_for_slot)
                elif roulette_candidate_pool:
                    logger.debug("Roulette pick fallback (should be rare): choosing random from pool.")
                    parents.append(random.choice(roulette_candidate_pool))

        final_parents = parents[:num_parents]
        logger.info(f"Total parents selected: {len(final_parents)}. IDs: {[p.id for p in final_parents]}")
        return final_parents

    def get_survivors(self, current_population: List[Program], offspring_population: List[Program],
                         population_size: int, task: TaskDefinition) -> List[Program]:
        logger.info(
            f"Starting survivor selection. Current pop: {len(current_population)}, Offspring pop: {len(offspring_population)}, Target pop_size: {population_size}")
        combined_population = current_population + offspring_population
        logger.debug(f"Combined population size for survivor selection: {len(combined_population)}")

        if not combined_population:
            logger.warning("Survivor selection called with empty combined population. Returning empty list.")
            return []

        sorted_combined_population = self.sort_programs(combined_population, task)

        survivors: List[Program] = []
        seen_ids_for_survivors = set()
        for program_in_sorted_list in sorted_combined_population:
            if len(survivors) < population_size:
                if program_in_sorted_list.id not in seen_ids_for_survivors:
                    survivors.append(program_in_sorted_list)
                    seen_ids_for_survivors.add(program_in_sorted_list.id)
            else:
                break

        logger.info(f"Selected {len(survivors)} survivors. IDs: {[p.id for p in survivors]}")
        return survivors

    async def execute(self, action: str, **kwargs) -> Any:
        task = kwargs.get('task')
        if not isinstance(task, TaskDefinition):
            raise ValueError("A TaskDefinition object must be provided in kwargs as 'task' for selection operations.")

        if action == "get_parents":
            population = kwargs.get('population')
            num_parents = kwargs.get('num_parents')
            if population is None or num_parents is None:
                raise ValueError("Missing 'population' or 'num_parents' for 'get_parents' action.")
            return self.get_parents(population, num_parents, task)
        elif action == "get_survivors":
            current_population = kwargs.get('current_population')
            offspring_population = kwargs.get('offspring_population')
            population_size = kwargs.get('population_size')
            if current_population is None or offspring_population is None or population_size is None:
                raise ValueError("Missing arguments for 'get_survivors' action.")
            return self.get_survivors(current_population, offspring_population, population_size, task)
        elif action == "sort_programs":
            programs_to_sort = kwargs.get('programs')
            if programs_to_sort is None:
                raise ValueError("Missing 'programs' for 'sort_programs' action.")
            return self.sort_programs(programs_to_sort, task)
        else:
            logger.error(f"Unknown action '{action}' for EvoSelector.execute().")
            raise ValueError(f"Unknown action: {action}")