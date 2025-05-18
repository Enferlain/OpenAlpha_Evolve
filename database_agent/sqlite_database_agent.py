# Version: 1.0.0 - Lumi's Decisive SQLite Agent!
import sqlite3
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Literal

from core.interfaces import DatabaseAgentInterface, Program, BaseAgent  # Assuming Program dataclass will have task_id
from config import settings

logger = logging.getLogger(__name__)

# Use the database path from settings, with a fallback.
DB_FILE_PATH = settings.DATABASE_PATH if settings.DATABASE_PATH and settings.DATABASE_PATH.endswith(
    ".db") else "alpha_evolve_programs.db"


class SQLiteDatabaseAgent(DatabaseAgentInterface, BaseAgent):
    """
    A DatabaseAgent that persists Program data to an SQLite database file.
    It uses asyncio.to_thread to run blocking SQLite operations in a separate thread,
    ensuring the asyncio event loop is not blocked. Each threaded operation
    manages its own database connection for thread-safety.
    """

    def __init__(self, db_file: Optional[str] = None):
        super().__init__()
        self.db_file = db_file if db_file else DB_FILE_PATH
        # No database connection is established in __init__.
        # Table creation is handled by the async setup_db method.
        logger.info(f"SQLiteDatabaseAgent initialized for database file: {self.db_file}. Call setup_db() to prepare.")

    def _execute_blocking_query(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False,
                                commit: bool = False):
        """
        Helper function to execute a blocking SQLite query in the current thread.
        This function is intended to be called via asyncio.to_thread().
        It establishes its own connection for each call to ensure thread safety.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row  # Access columns by name, so useful!
            cursor = conn.cursor()
            cursor.execute(query, params)

            if commit:
                conn.commit()
                return None  # Or return cursor.lastrowid or cursor.rowcount if needed

            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()

            return None  # Default for non-SELECT or non-commit operations if not specified
        except sqlite3.Error as e:
            logger.error(f"SQLite error during blocking query on {self.db_file}: {e}\nQuery: {query}\nParams: {params}",
                         exc_info=True)
            raise  # Re-raise so asyncio.to_thread can catch and propagate it
        finally:
            if conn:
                conn.close()

    async def _create_table_if_not_exists_async(self) -> None:  # v1.0.0
        """Creates the 'programs' table if it doesn't exist."""
        logger.info(f"Ensuring 'programs' table exists in SQLite database: {self.db_file}")
        create_table_query = """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                fitness_scores TEXT,
                generation INTEGER NOT NULL,
                parent_id TEXT,
                errors TEXT,
                status TEXT NOT NULL,
                task_id TEXT 
            )
        """
        # task_id is important for separating programs from different tasks!
        await asyncio.to_thread(
            self._execute_blocking_query,
            create_table_query,
            params=(),  # Explicitly provide default for params
            fetch_one=False,  # Explicitly provide default
            fetch_all=False,  # Explicitly provide default
            commit=True
        )
        logger.info(f"'programs' table ensured in {self.db_file}.")

    async def setup_db(self) -> None:  # v1.0.0
        """
        Asynchronously sets up the database by ensuring the necessary table exists.
        This should be called after initializing the agent and before any operations.
        """
        await self._create_table_if_not_exists_async()

    def _program_from_row(self, row: sqlite3.Row) -> Optional[Program]:  # v1.0.0
        """Converts a sqlite3.Row object to a Program object."""
        if not row:
            return None
        try:
            fitness_scores = json.loads(row["fitness_scores"]) if row["fitness_scores"] else {}
            errors = json.loads(row["errors"]) if row["errors"] else []

            return Program(
                id=row["id"],
                code=row["code"],
                fitness_scores=fitness_scores,
                generation=row["generation"],
                parent_id=row["parent_id"],
                errors=errors,
                status=row["status"],
                task_id=row["task_id"]
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for program ID {row['id']} from {self.db_file}: {e}", exc_info=True)
            return None
        except KeyError as e:
            logger.error(
                f"Missing expected key when creating Program from row for ID {row['id']} from {self.db_file}: {e}",
                exc_info=True)
            return None
        except Exception as e:  # Catch any other unexpected errors during reconstruction
            logger.error(f"Unexpected error reconstructing program ID {row['id']} from row from {self.db_file}: {e}",
                         exc_info=True)
            return None

    async def save_program(self, program: Program) -> None:  # v1.0.0
        """Saves a Program object to the SQLite database."""
        logger.debug(f"Attempting to save program ID: {program.id} to SQLite: {self.db_file}")

        if not program.task_id:
            logger.warning(
                f"Program ID {program.id} is being saved without a task_id. This might cause issues with retrieval later.")

        fitness_scores_json = json.dumps(program.fitness_scores)
        errors_json = json.dumps(program.errors)

        save_query = """
            INSERT OR REPLACE INTO programs 
            (id, code, fitness_scores, generation, parent_id, errors, status, task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            program.id,
            program.code,
            fitness_scores_json,
            program.generation,
            program.parent_id,
            errors_json,
            program.status,
            program.task_id
        )
        await asyncio.to_thread(
            self._execute_blocking_query,
            save_query,
            params,  # 'params' is already a variable here
            fetch_one=False,  # Explicitly provide default
            fetch_all=False,  # Explicitly provide default
            commit=True
        )
        logger.info(
            f"Successfully saved program ID: {program.id} (Task ID: {program.task_id}, Gen: {program.generation}) to SQLite: {self.db_file}")

    async def get_program(self, program_id: str) -> Optional[Program]:  # v1.0.0
        """Retrieves a single Program by its ID from the SQLite database."""
        logger.debug(f"Attempting to retrieve program by ID: {program_id} from SQLite: {self.db_file}")
        query = "SELECT * FROM programs WHERE id = ?"

        row = await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            (program_id,),
            fetch_one=True,
            fetch_all=False,  # Explicitly provide default
            commit=False  # Explicitly provide default
        )

        program = self._program_from_row(row)
        if program:
            logger.info(f"Retrieved program ID: {program.id} from SQLite: {self.db_file}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in SQLite database: {self.db_file}")
        return program

    async def get_all_programs(self) -> List[Program]:  # v1.0.0
        """Retrieves all programs from the SQLite database."""
        logger.debug(f"Attempting to retrieve all programs from SQLite: {self.db_file}")
        query = "SELECT * FROM programs"

        rows = await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            params=(),
            fetch_one=False,
            fetch_all=True,
            commit=False
        )

        programs = [self._program_from_row(row) for row in rows if row]
        # Filter out None results from _program_from_row if any errors occurred during reconstruction
        valid_programs = [p for p in programs if p is not None]
        logger.info(f"Retrieved {len(valid_programs)} programs from SQLite: {self.db_file}")
        return valid_programs

    async def get_best_programs(
            self,
            task_id: str,
            limit: int = 10,
            objective: Optional[str] = "correctness",  # Now using Optional[str] for flexibility
            sort_order: Literal["asc", "desc"] = "desc"
    ) -> List[Program]:  # v1.0.0
        """
        Retrieves the best programs for a given task_id, sorted by an objective.
        Sorting is performed in Python after fetching relevant programs.
        """
        logger.info(
            f"Retrieving best programs for task_id: {task_id}, limit: {limit}, objective: {objective}, order: {sort_order} from SQLite: {self.db_file}")

        # Fetch all programs for the given task_id first
        query = "SELECT * FROM programs WHERE task_id = ?"
        rows = await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            (task_id,),
            fetch_one=False,
            fetch_all=True,
            commit=False
        )

        all_task_programs = [self._program_from_row(row) for row in rows if row]
        valid_task_programs = [p for p in all_task_programs if p is not None]

        if not valid_task_programs:
            logger.info(f"No programs found for task_id: {task_id} in {self.db_file}")
            return []

        # Python-side sorting logic (consistent with InMemoryDatabaseAgent)
        def get_sort_key(program: Program):
            if objective == "correctness":
                return program.fitness_scores.get("correctness", -1.0)  # Higher is better
            elif objective == "runtime_ms":
                return program.fitness_scores.get("runtime_ms", float('inf'))  # Lower is better
            # Add other objectives from SelectionControllerAgent.METRIC_OPTIMIZATION_DIRECTION if needed
            # For example:
            # elif objective == "pylint_score":
            #     return program.fitness_scores.get("pylint_score", -10.0) # Higher is better
            # elif objective == "cyclomatic_complexity_avg":
            #     return program.fitness_scores.get("cyclomatic_complexity_avg", float('inf')) # Lower is better
            logger.warning(f"Unknown objective '{objective}' for sorting. Returning 0.")
            return 0  # Fallback for unknown objective

        # Determine reverse for sorted() based on objective and sort_order
        # True if higher score is better, False if lower score is better for the metric itself.
        metric_higher_is_better = True
        if objective == "runtime_ms" or objective == "cyclomatic_complexity_avg":  # Add other lower-is-better metrics
            metric_higher_is_better = False

        # If sort_order is 'desc', we want "better" values first.
        # If metric_higher_is_better is True (e.g. correctness), 'desc' means reverse=True.
        # If metric_higher_is_better is False (e.g. runtime), 'desc' (meaning worse runtimes first) means reverse=True.
        # This needs to be clearer:
        # sort_order 'desc' means: highest values of the key first.
        # sort_order 'asc' means: lowest values of the key first.

        # For 'correctness' (higher is better):
        #   'desc' -> sort by correctness descending (reverse=True)
        #   'asc'  -> sort by correctness ascending (reverse=False)
        # For 'runtime_ms' (lower is better):
        #   'desc' (worst runtimes first) -> sort by runtime descending (reverse=True, 100ms before 50ms)
        #   'asc'  (best runtimes first)  -> sort by runtime ascending (reverse=False, 50ms before 100ms)

        is_reverse_sort = (sort_order == "desc")

        sorted_programs = sorted(valid_task_programs, key=get_sort_key, reverse=is_reverse_sort)

        logger.debug(f"Sorted {len(sorted_programs)} programs for task_id: {task_id}. Returning top {limit}.")
        return sorted_programs[:limit]

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:  # v1.0.0
        """
        Selects a random sample of programs for a given task_id to seed the next generation.
        """
        logger.info(
            f"Attempting to retrieve {generation_size} random programs for task_id: {task_id} for next generation from SQLite: {self.db_file}")

        # SQLite's ORDER BY RANDOM() is efficient for this.
        query = "SELECT * FROM programs WHERE task_id = ? ORDER BY RANDOM() LIMIT ?"

        rows = await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            (task_id, generation_size),
            fetch_one=False,  # Correct
            fetch_all=True,  # Correct
            commit=False
        )

        programs = [self._program_from_row(row) for row in rows if row]
        valid_programs = [p for p in programs if p is not None]

        if len(valid_programs) < generation_size:
            logger.warning(
                f"Retrieved only {len(valid_programs)} programs for task_id: {task_id}, requested {generation_size}.")

        logger.info(
            f"Selected {len(valid_programs)} programs for next generation for task_id: {task_id} from {self.db_file}")
        return valid_programs

    async def count_programs(self, task_id: Optional[str] = None) -> int:  # v1.0.0
        """Counts programs, optionally filtered by task_id."""
        if task_id:
            logger.debug(f"Counting programs for task_id: {task_id} in SQLite: {self.db_file}")
            query = "SELECT COUNT(*) FROM programs WHERE task_id = ?"
            params = (task_id,)
        else:
            logger.debug(f"Counting all programs in SQLite: {self.db_file}")
            query = "SELECT COUNT(*) FROM programs"
            params = ()

        row = await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            params,
            fetch_one=True,
            fetch_all=False,
            commit=False
        )

        count = row[0] if row else 0
        logger.info(f"Found {count} programs (Task ID: {task_id if task_id else 'All'}) in SQLite: {self.db_file}")
        return count

    async def clear_database(self, task_id: Optional[str] = None) -> None:  # v1.0.0
        """Clears programs, optionally filtered by task_id."""
        if task_id:
            logger.info(f"Clearing programs for task_id: {task_id} from SQLite: {self.db_file}")
            query = "DELETE FROM programs WHERE task_id = ?"
            params = (task_id,)
        else:
            logger.info(f"Clearing ALL programs from SQLite: {self.db_file}. This is a destructive operation!")
            query = "DELETE FROM programs"
            params = ()

        await asyncio.to_thread(
            self._execute_blocking_query,
            query,
            params,
            fetch_one=False,
            fetch_all=False,
            commit=True
        )
        logger.info(f"Programs cleared (Task ID: {task_id if task_id else 'All'}) from SQLite: {self.db_file}")

    async def execute(self, *args, **kwargs) -> Any:  # v1.0.0
        logger.error(
            "SQLiteDatabaseAgent.execute() is not implemented. Use specific methods like save_program, get_program etc.")
        raise NotImplementedError(
            "SQLiteDatabaseAgent does not have a generic execute method. Please use specific database operations.")