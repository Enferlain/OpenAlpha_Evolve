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

    # --- NEW Field for LLM Judge Textual Feedback (Blueprint Requirement) ---
    llm_judge_feedback: Optional[str] = field(default=None)
    """Qualitative textual feedback from the LLM-as-a-judge."""

    # Note for EvaluatorAgent:
    # - `fitness_scores` will store 'llm_judge_overall_score', 'llm_judge_creativity_score' (and others from LLM judge)
    # - `fitness_scores` will also store 'runs_without_error' (bool, or 1.0/0.0 if strictly float needed for sorting keys)
    # - `fitness_scores` continues to store 'ruff_violations', 'runtime_ms', 'correctness' (if applicable), etc.

@dataclass
class TaskDefinition:
    # --- Core Inputs (Blueprint Requirement) ---
    id: str
    description: str # Rich, high-level problem/goal statement

    # --- Optional Inputs (Blueprint Requirements & Enhancements) ---
    initial_seed_code_or_ideas: Optional[str] = None # RENAMED from initial_seed_code, broader scope
    """A starting point, could be code, pseudocode, or even a list of concepts/ideas."""

    evaluation_guidelines_for_llm_judge: Optional[str] = None # NEW
    """Natural language criteria for an LLM to assess the solution's quality, effectiveness, creativity, etc."""

    sample_data_paths_or_env_description: Optional[str] = None # NEW
    """For tasks needing external data or a (mocked) interaction context (e.g., file paths, environment setup notes)."""

    suggested_imports: Optional[List[str]] = None # NEW (distinct from strictly enforced allowed_imports)
    """Optional hints for the LLM regarding useful Python libraries, not hard constraints."""

    target_solution_description: Optional[str] = None # NEW (replaces/generalizes function_name_to_evolve)
    """Describes what the final output should be (e.g., 'a Python script that...', 'a Python module containing class X')."""

    # --- Fields that become more "optional" or "hints" rather than strict constraints ---
    function_name_to_evolve: Optional[str] = None # Now more of a hint if target_solution_description is primary
    input_output_examples: Optional[List[Dict[str, Any]]] = None # Can be part of sample_data_paths or LLM judge's role
    allowed_imports: Optional[List[str]] = None # Becomes less strict, more like strong suggestions if suggested_imports is also used

    # --- Existing fields for general refinement / specific focus ---
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:" # Default prompt

    improvement_mode: Literal["task_focused", "general_refinement"] = "task_focused"
    """
    'task_focused': The agent focuses on the problem description and target_solution_description.
    'general_refinement': The agent focuses on improving 'initial_seed_code_or_ideas' based on general metrics
                          and 'specific_improvement_directives', potentially using I/O examples/LLM judge for regression/quality.
    """

    primary_focus_metrics: Optional[List[str]] = None
    """
    For 'general_refinement' or even 'task_focused' mode, list of metrics the LLM should primarily focus on improving
    (e.g., ["llm_judge_overall_score", "ruff_violations", "cyclomatic_complexity", "runtime_ms"]).
    This will be used by the EvaluatorAgent and PromptDesignerAgent.
    """

    specific_improvement_directives: Optional[str] = None
    """
    Natural language instructions, e.g., 'Focus on reducing runtime of the main loop.' or 'Refactor for better readability.'
    """

    evaluation_criteria: Optional[Dict[str, Any]] = None # General criteria, may inform LLM judge guidelines
    """e.g., {'target_metric': 'runtime_ms', 'goal': 'minimize'} - less primary if LLM judge is used."""


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
    async def manage_evolutionary_cycle(self):
        pass


class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def design_initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program,
                               evaluation_feedback: Optional[Dict] = None, # evaluation_feedback might be less direct now
                               ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:
        pass

    @abstractmethod
    def design_crossover_prompt(self, task: TaskDefinition, parent_program1: Program, parent_program2: Program) -> str:
        """Designs a prompt to guide the LLM in combining two parent programs."""
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_message: str,
                              execution_output: Optional[str] = None,
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:
        pass

    @abstractmethod
    def design_failed_diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,
                                           ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str:
        """Designs a prompt to ask for full code after a diff attempt failed."""
        pass

    @abstractmethod # NEW Method for LLM Judge Prompt
    def design_llm_judge_prompt(self,
                                task: TaskDefinition,
                                program_to_judge: Program,
                                execution_summary: Dict[str, Any] # Contains info like runs_ok, stdout, stderr, ruff_summary
                               ) -> str:
        """Designs a prompt to ask an LLM to act as a judge for the given program."""
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7,
                            output_format: str = "code") -> str:
        pass


class EvaluatorAgentInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass


class DatabaseAgentInterface(BaseAgent):
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
    def select_parents(self, evaluated_programs: List[Program], num_parents: int, task: TaskDefinition) -> List[Program]:
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int, task: TaskDefinition) -> List[Program]:
        pass

    @abstractmethod
    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[Program]:
        """Sorts a list of programs according to the agent's selection criteria."""
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass


class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict): # This will be called by log_generation_metrics or directly if needed
        pass

    @abstractmethod
    async def report_status(self): # May not be used if logging is continuous
        pass