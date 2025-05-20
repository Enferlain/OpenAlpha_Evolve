# database_agent/database_sqlite.py
# Version: 1.1.0 (Adding LLM Review Feedback persistence)

import sqlite3
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Literal

from core.interfaces import DatabaseInterface, Program, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

DB_FILE_PATH = settings.DATABASE_PATH if settings.DATABASE_PATH and settings.DATABASE_PATH.endswith(
    ".db") else "alpha_evolve_programs.db"


class SQLiteStore(DatabaseInterface, BaseAgent):
    def __init__(self, db_file: Optional[str] = None):
        super().__init__()
        self.db_file = db_file if db_file else DB_FILE_PATH
        logger.info(f"SQLiteStore initialized for database file: {self.db_file}. Call setup_db() to prepare.")

    def _execute_blocking_query(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False,
                                commit: bool = False):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
                return None
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error during blocking query on {self.db_file}: {e}\nQuery: {query}\nParams: {params}",
                         exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    # --- MODIFIED: _ensure_table (Blueprint Step 6) ---
    async def _ensure_table(self) -> None:  # Method_v1.1.0
        """Creates the 'programs' table if it doesn't exist, including ai_review_feedback."""
        logger.info(f"Ensuring 'programs' table exists in SQLite database: {self.db_file}")
        create_table_query = """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                fitness_scores TEXT,
                generation INTEGER NOT NULL,
                parent_id TEXT,
                parent_ids TEXT,
                errors TEXT,
                status TEXT NOT NULL,
                task_id TEXT,
                creation_method TEXT,
                ai_review_feedback TEXT  -- NEW COLUMN for LLM reviewer's textual feedback
            )
        """
        await asyncio.to_thread(
            self._execute_blocking_query,
            create_table_query,
            params=(), commit=True
        )
        logger.info(f"'programs' table ensured (with ai_review_feedback column) in {self.db_file}.")

    async def setup_db(self) -> None:
        await self._ensure_table()

    # --- MODIFIED: _program_from_row (Blueprint Step 6) ---
    def _program_from_row(self, row: sqlite3.Row) -> Optional[Program]:  # Method_v1.1.0
        if not row:
            return None
        try:
            fitness_scores = json.loads(row["fitness_scores"]) if row["fitness_scores"] else {}
            errors = json.loads(row["errors"]) if row["errors"] else []
            parent_ids_list = json.loads(row["parent_ids"]) if row["parent_ids"] else None

            # Ensure all expected keys are present in the row object
            # This uses .get() with a default for the new column to be robust
            # if loading from an older DB schema that doesn't have it yet (though CREATE TABLE adds it).
            ai_review_feedback_value = row["ai_review_feedback"] if "ai_review_feedback" in row.keys() else None

            return Program(
                id=row["id"],
                code=row["code"],
                fitness_scores=fitness_scores,
                generation=row["generation"],
                parent_id=row["parent_id"],
                parent_ids=parent_ids_list,
                errors=errors,
                status=row["status"],
                task_id=row["task_id"],
                creation_method=row["creation_method"] if "creation_method" in row.keys() else "unknown",
                ai_review_feedback=ai_review_feedback_value  # NEW: Load the feedback
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for program ID {row['id']} from {self.db_file}: {e}", exc_info=True)
            return None
        except KeyError as e:
            logger.error(
                f"Missing expected key when creating Program from row for ID {row['id']} from {self.db_file}: {e}. Row keys: {row.keys()}",
                exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error reconstructing program ID {row['id']} from row from {self.db_file}: {e}",
                         exc_info=True)
            return None

    # --- MODIFIED: save_program (Blueprint Step 6) ---
    async def save_program(self, program: Program) -> None:  # Method_v1.1.0
        logger.debug(f"Attempting to save program ID: {program.id} to SQLite: {self.db_file}")

        if not program.task_id:
            logger.warning(
                f"Program ID {program.id} is being saved without a task_id. This might cause issues with retrieval later.")

        fitness_scores_json = json.dumps(program.fitness_scores)
        errors_json = json.dumps(program.errors)
        parent_ids_json = json.dumps(program.parent_ids) if program.parent_ids is not None else None

        # Ensure ai_review_feedback is None if not set, not an empty string from Program default if that changed.
        # Program dataclass default is None, so this should be fine.
        ai_review_feedback_to_save = program.ai_review_feedback

        save_query = """
            INSERT OR REPLACE INTO programs 
            (id, code, fitness_scores, generation, parent_id, parent_ids, errors, status, task_id, creation_method, ai_review_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            program.id,
            program.code,
            fitness_scores_json,
            program.generation,
            program.parent_id,
            parent_ids_json,
            errors_json,
            program.status,
            program.task_id,
            program.creation_method,
            ai_review_feedback_to_save  # NEW: Save the feedback
        )
        await asyncio.to_thread(
            self._execute_blocking_query,
            save_query, params, commit=True
        )
        logger.info(
            f"Successfully saved program ID: {program.id} (Task ID: {program.task_id}, Gen: {program.generation}) to SQLite: {self.db_file}")

    # --- get_program, get_all_programs, get_best_programs, get_programs_for_next_generation, count_programs, clear_database, execute ---
    # These methods do not need to change for this specific update, as they rely on _program_from_row
    # or operate on all columns / don't directly interact with ai_review_feedback beyond what _program_from_row provides.
    # (The existing implementations of these methods will be used)

    async def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program by ID: {program_id} from SQLite: {self.db_file}")
        query = "SELECT * FROM programs WHERE id = ?"
        row = await asyncio.to_thread(
            self._execute_blocking_query, query, (program_id,), fetch_one=True
        )
        program = self._program_from_row(row)  # Will now include ai_review_feedback
        if program:
            logger.info(f"Retrieved program ID: {program.id} from SQLite: {self.db_file}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in SQLite database: {self.db_file}")
        return program

    async def get_all_programs(self) -> List[Program]:
        logger.debug(f"Attempting to retrieve all programs from SQLite: {self.db_file}")
        query = "SELECT * FROM programs"
        rows = await asyncio.to_thread(self._execute_blocking_query, query, fetch_all=True)
        programs = [self._program_from_row(row) for row in rows if row]
        valid_programs = [p for p in programs if p is not None]
        logger.info(f"Retrieved {len(valid_programs)} programs from SQLite: {self.db_file}")
        return valid_programs

    # For get_best_programs, the sorting happens in Python based on fitness_scores, which will
    # now naturally include ai_review_score if SelectionController uses it.
    async def get_best_programs(
            self,
            task_id: str,
            limit: int = 10,
            objective: Optional[str] = "correctness",
            sort_order: Literal["asc", "desc"] = "desc"
    ) -> List[Program]:
        logger.info(
            f"Retrieving best programs for task_id: {task_id}, limit: {limit}, objective: {objective}, order: {sort_order} from SQLite: {self.db_file}")
        query = "SELECT * FROM programs WHERE task_id = ?"
        rows = await asyncio.to_thread(
            self._execute_blocking_query, query, (task_id,), fetch_all=True
        )
        all_task_programs = [self._program_from_row(row) for row in rows if row]
        valid_task_programs = [p for p in all_task_programs if p is not None]

        if not valid_task_programs:
            logger.info(f"No programs found for task_id: {task_id} in {self.db_file}")
            return []

        def get_sort_key(program: Program):
            # This logic should ideally mirror or use SelectionController's logic if it's complex
            # For now, simple objective based on common metrics
            score = program.fitness_scores.get(objective, 0.0)
            if objective in ["runtime_ms", "ruff_violations", "cyclomatic_complexity_avg"]: # Lower is better
                return score if score is not None else float('inf')
            return score if score is not None else 0.0 # Higher is better

        is_reverse_sort = (sort_order == "desc")
        # Adjust reverse if lower is better for the metric
        if objective in ["runtime_ms", "ruff_violations", "cyclomatic_complexity_avg"]:
            is_reverse_sort = not is_reverse_sort # if sort_order is desc (worst first), then reverse=True. if asc (best first), reverse=False.

        sorted_programs = sorted(valid_task_programs, key=get_sort_key, reverse=is_reverse_sort)
        logger.debug(f"Sorted {len(sorted_programs)} programs for task_id: {task_id}. Returning top {limit}.")
        return sorted_programs[:limit]

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        logger.info(
            f"Attempting to retrieve {generation_size} random programs for task_id: {task_id} for next generation from SQLite: {self.db_file}")
        query = "SELECT * FROM programs WHERE task_id = ? ORDER BY RANDOM() LIMIT ?"
        rows = await asyncio.to_thread(
            self._execute_blocking_query, query, (task_id, generation_size), fetch_all=True
        )
        programs = [self._program_from_row(row) for row in rows if row]
        valid_programs = [p for p in programs if p is not None]
        if len(valid_programs) < generation_size:
            logger.warning(
                f"Retrieved only {len(valid_programs)} programs for task_id: {task_id}, requested {generation_size}.")
        logger.info(
            f"Selected {len(valid_programs)} programs for next generation for task_id: {task_id} from {self.db_file}")
        return valid_programs

    async def count_programs(self, task_id: Optional[str] = None) -> int:
        if task_id:
            logger.debug(f"Counting programs for task_id: {task_id} in SQLite: {self.db_file}")
            query = "SELECT COUNT(*) FROM programs WHERE task_id = ?"
            params = (task_id,)
        else:
            logger.debug(f"Counting all programs in SQLite: {self.db_file}")
            query = "SELECT COUNT(*) FROM programs"
            params = ()
        row = await asyncio.to_thread(
            self._execute_blocking_query, query, params, fetch_one=True
        )
        count = row[0] if row else 0
        logger.info(f"Found {count} programs (Task ID: {task_id if task_id else 'All'}) in SQLite: {self.db_file}")
        return count

    async def clear_database(self, task_id: Optional[str] = None) -> None:
        if task_id:
            logger.info(f"Clearing programs for task_id: {task_id} from SQLite: {self.db_file}")
            query = "DELETE FROM programs WHERE task_id = ?"
            params = (task_id,)
        else:
            logger.info(f"Clearing ALL programs from SQLite: {self.db_file}. This is a destructive operation!")
            query = "DELETE FROM programs"
            params = ()
        await asyncio.to_thread(
            self._execute_blocking_query, query, params, commit=True
        )
        logger.info(f"Programs cleared (Task ID: {task_id if task_id else 'All'}) from SQLite: {self.db_file}")

    async def execute(self, *args, **kwargs) -> Any:
        logger.error(
            "SQLiteStore.execute() is not implemented. Use specific methods like save_program, get_program etc.")
        raise NotImplementedError(
            "SQLiteStore does not have a generic execute method. Please use specific database operations.")