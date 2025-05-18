import random
import logging
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# We need TaskDefinition if we're going to make decisions based on it.
from core.interfaces import SelectionControllerInterface, Program, BaseAgent, TaskDefinition # Added TaskDefinition_v2
from config import settings

logger = logging.getLogger(__name__)

# --- Helper for metric optimization direction (v2) ---
# True if higher score is better, False if lower score is better
METRIC_OPTIMIZATION_DIRECTION = {
    "correctness": True,
    "pylint_score": True,
    "maintainability_index": True,
    "passed_tests": True, # More passed tests is better
    "runtime_ms": False,
    "cyclomatic_complexity_avg": False,
    # Add other metrics here as they are introduced
}
DEFAULT_METRIC_VALUE = {
    "correctness": 0.0,
    "pylint_score": -1.0, # Pylint usually 0-10
    "maintainability_index": -1.0, # MI usually 0-100
    "passed_tests": 0.0,
    "runtime_ms": float('inf'),
    "cyclomatic_complexity_avg": float('inf'),
}
# ---


class SelectionControllerAgent(SelectionControllerInterface, BaseAgent):
    def __init__(self):  # Unchanged
        super().__init__()
        self.elitism_count = settings.ELITISM_COUNT
        logger.info(f"SelectionControllerAgent initialized with elitism_count: {self.elitism_count}")

    def _get_program_sort_key(self, program: Program, task: TaskDefinition) -> Tuple:  # New helper method _v2
        """
        Creates a sort key for a program based on the task definition.
        Sorts primarily by I/O correctness, then by primary_focus_metrics, then by default tie-breakers.
        """
        key_parts = []

        # 1. Correctness from I/O examples (always important if available)
        # Higher is better, so for ascending sort, we'd use -correctness if sorting everything ascending.
        # Or, rely on reverse=True later and keep it positive. Let's use positive for now.
        correctness = program.fitness_scores.get("correctness", DEFAULT_METRIC_VALUE["correctness"])
        key_parts.append(correctness)

        # 2. Primary Focus Metrics from TaskDefinition (if in "general_refinement" or specified)
        if task.primary_focus_metrics:
            for metric_name in task.primary_focus_metrics:
                value = program.fitness_scores.get(metric_name, DEFAULT_METRIC_VALUE.get(metric_name, 0.0))
                higher_is_better = METRIC_OPTIMIZATION_DIRECTION.get(metric_name, True)  # Default to True if unknown

                if higher_is_better:
                    key_parts.append(value)  # Will be sorted descending by reverse=True
                else:
                    key_parts.append(-value)  # Lower is better, negate to make it sort correctly with reverse=True
                    # e.g., -50ms is "better" than -100ms when sorted descending.
                    # No, if reverse=True, lower is better should be added as is, higher as negative.
                    # Let's rethink: if reverse=True, we want bigger numbers first.
                    # If higher is better (correctness=0.9), it's fine.
                    # If lower is better (runtime=50), we want 50 to come before 100.
                    # So if reverse=True, for "lower is better" metrics, we need them to be smaller.
                    # So, we use -metric_value for "lower is better" to make smaller actuals appear "larger" in the key for reverse sort.
                    # No, this is confusing. Let's simplify:
                    # Python's sort is stable. We build a tuple.
                    # For reverse=True:
                    #   - "higher is better": use value directly (e.g., 0.9 before 0.8)
                    #   - "lower is better": use -value (e.g., -50 before -100, so 50ms before 100ms)
                    key_parts.append(value if higher_is_better else -value)

        # 3. Default tie-breakers (e.g., runtime if not a primary focus, then generation)
        if "runtime_ms" not in (task.primary_focus_metrics or []):  # If runtime wasn't primary
            runtime = program.fitness_scores.get("runtime_ms", DEFAULT_METRIC_VALUE["runtime_ms"])
            key_parts.append(-runtime)  # Lower is better, so -runtime for descending sort.

        key_parts.append(
            program.generation)  # Favor newer generations slightly in case of perfect ties so far (higher gen is better)

        # logger.debug(f"Program {program.id} sort key: {key_parts}")
        return tuple(key_parts)

    # select_parents now takes task_definition
    def select_parents(self, population: List[Program], num_parents: int, task: TaskDefinition) -> List[
        Program]:  # Added task_v2
        logger.info(
            f"Starting parent selection. Pop size: {len(population)}, Num parents: {num_parents}, Mode: {task.improvement_mode}")
        if not population:
            logger.warning("Parent selection called with empty population. Returning empty list.")
            return []
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []

        # Sort population by fitness (using the new dynamic key)
        # reverse=True means programs with "better" (larger) key tuples come first.
        sorted_population = sorted(
            population,
            key=lambda p: self._get_program_sort_key(p, task),
            reverse=True
        )
        top_program_ids_and_keys = [
            (p.id, p.fitness_scores.get("correctness", 0), p.fitness_scores.get("pylint_score", 0),
             self._get_program_sort_key(p, task)) for p in sorted_population[:5]
        ]
        logger.debug(
            f"Population sorted for parent selection. Top 5 (IDs, Correctness, Pylint, FullKey): {top_program_ids_and_keys}")

        parents = []
        # 1. Elitism: Select the top N unique individuals
        elite_candidates = []
        seen_ids_for_elitism = set()
        for program in sorted_population:
            if len(elite_candidates) < self.elitism_count:
                if program.id not in seen_ids_for_elitism:
                    elite_candidates.append(program)
                    seen_ids_for_elitism.add(program.id)
            else:
                break
        parents.extend(elite_candidates)
        logger.info(f"Selected {len(elite_candidates)} elite parents: {[p.id for p in elite_candidates]}")

        remaining_slots = num_parents - len(parents)
        if remaining_slots <= 0:
            logger.info("Elitism filled all parent slots or no more parents needed.")
            return parents

        roulette_candidates = [p for p in sorted_population if p.id not in seen_ids_for_elitism]
        if not roulette_candidates:
            logger.warning("No candidates left for roulette selection after elitism. Returning current parents.")
            return parents

        # Fitness-Proportionate Selection (Roulette Wheel)
        # For roulette, we typically need a single positive fitness value.
        # We can use the 'correctness' score primarily, or a composite if in general_refinement.
        # For simplicity, let's stick to 'correctness' for roulette for now,
        # as elitism already picked the multi-objective best.
        # This part might need more thought for a truly multi-objective roulette.
        total_fitness_roulette = sum(p.fitness_scores.get("correctness", 0.0) + 0.0001 for p in roulette_candidates)
        logger.debug(
            f"Total 'correctness' fitness for roulette wheel (among {len(roulette_candidates)} candidates): {total_fitness_roulette:.4f}")

        if total_fitness_roulette <= 0.0001 * len(roulette_candidates):
            logger.warning("All roulette candidates have near-zero correctness. Selecting randomly from them.")
            num_to_select_randomly = min(remaining_slots, len(roulette_candidates))
            random_parents = random.sample(roulette_candidates, num_to_select_randomly)
            parents.extend(random_parents)
            logger.info(
                f"Selected {len(random_parents)} parents randomly due to zero total correctness: {[p.id for p in random_parents]}")
        else:
            for _ in range(remaining_slots):
                if not roulette_candidates: break
                pick = random.uniform(0, total_fitness_roulette)
                current_sum = 0
                chosen_parent = None
                for program in roulette_candidates:
                    current_sum += (program.fitness_scores.get("correctness", 0.0) + 0.0001)
                    if current_sum >= pick:
                        chosen_parent = program
                        break
                if chosen_parent:
                    parents.append(chosen_parent)
                else:  # Fallback
                    if roulette_candidates:
                        parents.append(random.choice(roulette_candidates))

        logger.info(f"Total parents selected: {len(parents)}. IDs: {[p.id for p in parents]}")
        return parents

    # select_survivors now takes task_definition
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program],
                         population_size: int, task: TaskDefinition) -> List[Program]:  # Added task_v2
        logger.info(
            f"Starting survivor selection. Current: {len(current_population)}, Offspring: {len(offspring_population)}, Target: {population_size}, Mode: {task.improvement_mode}")
        combined_population = current_population + offspring_population
        logger.debug(f"Combined population size for survivor selection: {len(combined_population)}")

        if not combined_population:
            logger.warning("Survivor selection called with empty combined population. Returning empty list.")
            return []

        # Sort by the dynamic fitness key
        sorted_combined = sorted(
            combined_population,
            key=lambda p: self._get_program_sort_key(p, task),
            reverse=True  # "Better" (larger) key tuples come first
        )

        top_program_ids_and_keys_survivor = [
            (p.id, p.fitness_scores.get("correctness", 0), p.fitness_scores.get("pylint_score", 0),
             self._get_program_sort_key(p, task)) for p in sorted_combined[:5]
        ]
        logger.debug(
            f"Combined population sorted for survivor selection. Top 5 (IDs, Correctness, Pylint, FullKey): {top_program_ids_and_keys_survivor}")

        survivors = []
        seen_program_ids = set()  # Ensure unique programs in the new generation
        for program in sorted_combined:
            if len(survivors) < population_size:
                if program.id not in seen_program_ids:  # IDs should be unique anyway if generated properly
                    survivors.append(program)
                    seen_program_ids.add(program.id)
            else:
                break

        logger.info(f"Selected {len(survivors)} survivors. IDs: {[p.id for p in survivors]}")
        return survivors

    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[Program]: # New method _v3
        if not programs:
            return []
        return sorted(
            programs,
            key=lambda p: self._get_program_sort_key(p, task),
            reverse=True
        )

    # The execute method will now need task_definition passed in kwargs if this agent is called via TaskManager.execute
    # However, TaskManagerAgent calls select_parents and select_survivors directly.
    async def execute(self, action: str,
                      **kwargs) -> Any:  # Unchanged signature, but caller needs to provide 'task' in kwargs
        task = kwargs.get('task')
        if not isinstance(task, TaskDefinition):
            raise ValueError("A TaskDefinition object must be provided in kwargs as 'task' for selection operations.")

        if action == "select_parents":
            return self.select_parents(kwargs['population'], kwargs['num_parents'], task)
        elif action == "select_survivors":
            return self.select_survivors(kwargs['current_population'], kwargs['offspring_population'],
                                         kwargs['population_size'], task)
        else:
            raise ValueError(f"Unknown action: {action}")

# Example Usage:
if __name__ == '__main__':
    import uuid
    logging.basicConfig(level=logging.DEBUG)
    selector = SelectionControllerAgent()

    # Create some sample programs
    programs = [
        Program(id=str(uuid.uuid4()), code="c1", fitness_scores={"correctness": 0.9, "runtime_ms": 100}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c2", fitness_scores={"correctness": 1.0, "runtime_ms": 50}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c3", fitness_scores={"correctness": 0.7, "runtime_ms": 200}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c4", fitness_scores={"correctness": 1.0, "runtime_ms": 60}, status="evaluated"), # Duplicate high correctness
        Program(id=str(uuid.uuid4()), code="c5", fitness_scores={"correctness": 0.5}, status="evaluated"), # Missing runtime
        Program(id=str(uuid.uuid4()), code="c6", status="unevaluated"), # Unevaluated
    ]

    print("--- Testing Parent Selection ---")
    parents = selector.select_parents(programs, num_parents=3)
    for p in parents:
        print(f"Selected Parent: {p.id}, Correctness: {p.fitness_scores.get('correctness')}, Runtime: {p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Survivor Selection ---")
    current_pop = programs[:2] # p1, p2
    offspring_pop = [
        Program(id=str(uuid.uuid4()), code="off1", fitness_scores={"correctness": 1.0, "runtime_ms": 40}, status="evaluated"), # Better than p2
        Program(id=str(uuid.uuid4()), code="off2", fitness_scores={"correctness": 0.6, "runtime_ms": 10}, status="evaluated"),
    ]
    survivors = selector.select_survivors(current_pop, offspring_pop, population_size=2)
    for s in survivors:
        print(f"Survivor: {s.id}, Correctness: {s.fitness_scores.get('correctness')}, Runtime: {s.fitness_scores.get('runtime_ms')}") 