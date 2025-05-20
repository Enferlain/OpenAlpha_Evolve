import logging
from typing import List, Dict, Any, Optional, Literal
import uuid

from core.interfaces import (
    DatabaseInterface,
    Program,
    BaseAgent,
)
# from config import settings # Settings not used in this agent

logger = logging.getLogger(__name__)

class InMemoryStore(DatabaseInterface, BaseAgent):
    """An in-memory database for storing and retrieving programs."""
    def __init__(self):
        super().__init__()
        self._programs: Dict[str, Program] = {}
        logger.info("InMemoryStore initialized.")

    async def save_program(self, program: Program) -> None:
        logger.info(f"Saving program: {program.id} (Generation: {program.generation}) to in-memory database.")
        if program.id in self._programs:
            logger.warning(f"Program with ID {program.id} already exists. It will be overwritten.")
        self._programs[program.id] = program
        logger.debug(f"Program {program.id} data: {program}")

    async def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program by ID: {program_id}")
        program = self._programs.get(program_id)
        if program:
            logger.info(f"Retrieved program: {program.id}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in database.")
        return program

    async def get_all_programs(self) -> List[Program]:
        logger.debug(f"Retrieving all {len(self._programs)} programs from in-memory database.")
        return list(self._programs.values())

    async def get_best_programs(
        self,
        task_id: str,
        limit: int = 5,
        objective: Literal["correctness", "runtime_ms"] = "correctness",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> List[Program]:
        logger.info(f"Retrieving best programs for task {task_id}. Limit: {limit}, Objective: {objective}, Order: {sort_order}")
        if not self._programs:
            logger.info("No programs in database to retrieve 'best' from.")
            return []

        all_progs = list(self._programs.values())
        # Placeholder: If programs were associated with task_ids, filter here:
        # relevant_progs = [p for p in all_progs if p.task_id == task_id] # Assuming Program has a task_id field
        # For now, using all_progs as task_id is not stored/filtered on Program objects in this simple DB.
        relevant_progs = all_progs

        def sort_key(p: Program):
            if objective == "correctness":
                return p.fitness_scores.get("correctness", -1.0)
            elif objective == "runtime_ms":
                val = p.fitness_scores.get("runtime_ms", float('inf'))
                # For runtime_ms, if sort_order is 'desc', we want higher values first (worse runtimes).
                # If sort_order is 'asc', we want lower values first (better runtimes).
                # The key should return a value that, when sorted in ascending order by default by `sorted`,
                # achieves the desired `sort_order`.
                # So if we want descending, the key should be negated for numeric types.
                return -val if sort_order == "desc" else val 
            return 0 # Should not happen with Literal types

        # Determine reverse for sorted() based on objective and sort_order
        if objective == "correctness": # Higher correctness is better
            effective_reverse = (sort_order == "desc")
        elif objective == "runtime_ms": # Lower runtime_ms is better
            effective_reverse = (sort_order == "asc") # sort ascending for runtime means reverse=False
        else:
            effective_reverse = False # Default
        
        # Corrected sorting logic based on objective and sort_order
        if objective == "runtime_ms":
            # Ascending runtime_ms (better) means reverse=False for sorted()
            # Descending runtime_ms (worse) means reverse=True for sorted()
            sorted_programs = sorted(relevant_progs, key=lambda p: p.fitness_scores.get("runtime_ms", float('inf')), reverse=(sort_order == "desc"))
        elif objective == "correctness":
            # Ascending correctness (worse) means reverse=False
            # Descending correctness (better) means reverse=True
            sorted_programs = sorted(relevant_progs, key=lambda p: p.fitness_scores.get("correctness", -1.0), reverse=(sort_order == "desc"))
        else:
            # Fallback, though Literal should prevent this.
            sorted_programs = sorted(relevant_progs, key=sort_key, reverse=effective_reverse)

        logger.debug(f"Sorted {len(sorted_programs)} programs. Top 3 (if available): {[p.id for p in sorted_programs[:3]]}")
        return sorted_programs[:limit]

    async def get_programs_by_generation(self, generation: int) -> List[Program]:
        logger.debug(f"Retrieving programs for generation: {generation}")
        generation_programs = [p for p in self._programs.values() if p.generation == generation]
        logger.info(f"Found {len(generation_programs)} programs for generation {generation}.")
        return generation_programs

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        logger.info(f"Attempting to retrieve {generation_size} programs for next generation for task {task_id}.")
        all_progs = list(self._programs.values())
        if not all_progs:
            logger.warning("No programs in database to select for next generation.")
            return []

        if len(all_progs) <= generation_size:
            logger.debug(f"Returning all {len(all_progs)} programs as it's less than or equal to generation_size {generation_size}.")
            return all_progs
        
        import random
        selected_programs = random.sample(all_progs, generation_size)
        logger.info(f"Selected {len(selected_programs)} random programs for next generation.")
        return selected_programs

    async def count_programs(self) -> int:
        count = len(self._programs)
        logger.debug(f"Total programs in database: {count}")
        return count

    async def clear_database(self) -> None:
        logger.info("Clearing all programs from in-memory database.")
        self._programs.clear()
        logger.info("In-memory database cleared.")

    async def execute(self, *args, **kwargs) -> Any:
        logger.warning("InMemoryStore.execute() called, but this agent uses specific methods for DB operations.")
        raise NotImplementedError("InMemoryStore does not have a generic execute. Use specific methods like save_program, get_program etc.")