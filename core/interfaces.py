# core/interfaces.py
# Version: 3.0.0 (Blueprint for Emergent AI Discovery Integration)

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field


@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, Any] = field(default_factory=dict) # Changed to Any to accommodate bool for runs_without_error
    generation: int = 0
    parent_id: Optional[str] = None
    parent_ids: Optional[List[str]] = None
    errors: List[str] = field(default_factory=list) # Will store Ruff issues, execution errors
    status: str = "unevaluated"
    task_id: Optional[str] = None
    creation_method: str = "unknown"

    # --- NEW Field for LLM Review Textual Feedback (Blueprint Requirement) ---
    ai_review_feedback: Optional[str] = field(default=None)
    """Qualitative textual feedback from the LLM-as-a-reviewer."""

    # Note for SolutionEvaluator:
    # - `fitness_scores` will store 'ai_review_score', 'ai_review_creativity_score' (and others from LLM review)
    # - `fitness_scores` will also store 'runs_without_error' (bool, or 1.0/0.0 if strictly float needed for sorting keys)
    # - `fitness_scores` continues to store 'ruff_violations', 'runtime_ms', 'correctness' (if applicable), etc.

@dataclass
class TaskDefinition:
    # --- Core Inputs (Blueprint Requirement) ---
    id: str
    description: str # Rich, high-level problem/goal statement

    # --- Optional Inputs (Blueprint Requirements & Enhancements) ---
    initial_seed: Optional[str] = None # RENAMED from initial_seed_code, broader scope
    """A starting point, could be code, pseudocode, or even a list of concepts/ideas."""

    ai_review_criteria: Optional[str] = None # NEW
    """Natural language criteria for an LLM to assess the solution's quality, effectiveness, creativity, etc."""

    run_context: Optional[str] = None # NEW
    """For tasks needing external data or a (mocked) interaction context (e.g., file paths, environment setup notes)."""

    suggested_imports: Optional[List[str]] = None # NEW (distinct from strictly enforced allowed_imports)
    """Optional hints for the LLM regarding useful Python libraries, not hard constraints."""

    target_solution: Optional[str] = None # NEW (replaces/generalizes evolve_function)
    """Describes what the final output should be (e.g., 'a Python script that...', 'a Python module containing class X')."""

    # --- Fields that become more "optional" or "hints" rather than strict constraints ---
    evolve_function: Optional[str] = None # Now more of a hint if target_solution is primary
    io_examples: Optional[List[Dict[str, Any]]] = None # Can be part of sample_data_paths or LLM reviewer's role
    allowed_imports: Optional[List[str]] = None # Becomes less strict, more like strong suggestions if suggested_imports is also used

    # --- Existing fields for general refinement / specific focus ---
    initial_prompt_override: Optional[str] = "Provide an initial Python solution for the following problem:" # Default prompt

    improvement_mode: Literal["task_focused", "general_refinement"] = "task_focused"
    """
    'task_focused': The agent focuses on the problem description and target_solution.
    'general_refinement': The agent focuses on improving 'initial_seed' based on general metrics
                          and 'refine_goals', potentially using I/O examples/LLM review for regression/quality.
    """

    primary_focus_metrics: Optional[List[str]] = None
    """
    For 'general_refinement' or even 'task_focused' mode, list of metrics the LLM should primarily focus on improving
    (e.g., ["ai_review_score", "ruff_violations", "cyclomatic_complexity", "runtime_ms"]).
    This will be used by the SolutionEvaluator and PromptStudio.
    """

    refine_goals: Optional[str] = None
    """
    Natural language instructions, e.g., 'Focus on reducing runtime of the main loop.' or 'Refactor for better readability.'
    """

    evaluation_criteria: Optional[Dict[str, Any]] = None # General criteria, may inform ai review guidelines
    """e.g., {'target_metric': 'runtime_ms', 'goal': 'minimize'} - less primary if LLM review is used."""


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Main execution method for an agent."""
        pass


class TaskManagerInterface(BaseAgent):
    @abstractmethod
    async def run_evolution(self):
        pass


class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def mutation_prompt(self,
                               task: TaskDefinition,
                               parent_program: Program, # Main source of feedback
                               ancestral_summary: Optional[List[Dict[str, Any]]] = None
                               ) -> str:
        pass

    @abstractmethod
    def crossover_prompt(self, task: TaskDefinition, parent_program1: Program, parent_program2: Program) -> str:
        """Designs a prompt to guide the LLM in combining two parent programs."""
        pass

    @abstractmethod
    def bugfix_prompt(self,
                              task: TaskDefinition,
                              program: Program, # Main source of error/feedback context
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None
                              ) -> str:
        pass

    @abstractmethod
    def diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,
                                           ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:
        """Designs a prompt to ask for full code after a diff attempt failed."""
        pass

    @abstractmethod # NEW Method for ai review Prompt
    def ai_review_prompt(self,
                                task: TaskDefinition,
                                program_to_review: Program,
                                execution_summary: Dict[str, Any] # Contains info like runs_ok, stdout, stderr, ruff_summary
                               ) -> str:
        """Designs a prompt to ask an LLM to act as a reviewer for the given program."""
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7,
                            output_format: str = "code") -> str:
        pass


class SolutionEvaluatorInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass


class DatabaseInterface(BaseAgent):
    @abstractmethod
    async def save_program(self, program: Program):
        pass

    @abstractmethod
    async def get_program(self, program_id: str) -> Optional[Program]:
        pass

    @abstractmethod
    async def get_all_programs(self) -> List[Program]:
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass

    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass


class SelectionControllerInterface(BaseAgent):
    @abstractmethod
    def get_parents(self, evaluated_programs: List[Program], num_parents: int, task: TaskDefinition) -> List[Program]:
        pass

    @abstractmethod
    def get_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int, task: TaskDefinition) -> List[Program]:
        pass

    @abstractmethod
    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[Program]:
        """Sorts a list of programs according to the agent's selection criteria."""
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass


class MetricsLoggerInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict): # This will be called by log_generation_metrics or directly if needed
        pass

    @abstractmethod
    async def report_status(self): # May not be used if logging is continuous
        pass