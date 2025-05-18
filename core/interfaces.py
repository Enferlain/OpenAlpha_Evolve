# Core components, interfaces, data models 
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal # Added Literal_v2
from dataclasses import dataclass, field

@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None # For mutation, could store one parent
    parent_ids: Optional[List[str]] = None # For crossover, could store both parents!
    errors: List[str] = field(default_factory=list)
    status: str = "unevaluated"
    task_id: Optional[str] = None
    creation_method: str = "unknown" # e.g., "initial", "mutation", "bug_fix", "crossover" <-- NEW!

@dataclass
class TaskDefinition:
    id: str
    description: str # Natural language description of the problem
    function_name_to_evolve: Optional[str] = None # Name of the function the LLM should generate/evolve
    input_output_examples: Optional[List[Dict[str, Any]]] = None # For testing, e.g. [{"input": ..., "output": ...}]
    evaluation_criteria: Optional[Dict[str, Any]] = None # e.g., {"target_metric": "runtime_ms", "goal": "minimize"}
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:"
    allowed_imports: Optional[List[str]] = None

    # --- New fields for objective-agnostic improvement (v2) ---
    initial_seed_code: Optional[str] = None
    """If provided, this code will be used as the starting point for the first program in the population."""

    improvement_mode: Literal["task_focused", "general_refinement"] = "task_focused"
    """
    'task_focused': The agent focuses on the problem description and I/O examples.
    'general_refinement': The agent focuses on improving the 'initial_seed_code' based on general metrics 
                          and 'specific_improvement_directives', potentially still using I/O examples for regression.
    """

    primary_focus_metrics: Optional[List[str]] = None
    """
    For 'general_refinement' mode, list of metrics the LLM should primarily focus on improving 
    (e.g., ["pylint_score", "cyclomatic_complexity", "runtime_ms"]).
    This will be used by the EvaluatorAgent and PromptDesignerAgent later.
    """

    specific_improvement_directives: Optional[str] = None
    """
    Natural language instructions for 'general_refinement' mode, e.g., 
    'Focus on reducing runtime of the main loop.' or 'Refactor for better readability.'
    """
    # --- End of new fields (v2) ---


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
    def design_initial_prompt(self, task: TaskDefinition) -> str: # Should be correct now
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program,
                               evaluation_feedback: Optional[Dict] = None,
                               ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str: # Method_v2 (added ancestral_summary)
        pass

    @abstractmethod
    def design_crossover_prompt(self, task: TaskDefinition, parent_program1: Program, parent_program2: Program) -> str: # <-- NEW!
        """Designs a prompt to guide the LLM in combining two parent programs."""
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_message: str,
                              execution_output: Optional[str] = None,
                              ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str: # Method_v6 (added ancestral_summary)
        pass

    @abstractmethod
    def design_failed_diff_fallback_prompt(self, task: TaskDefinition, original_program: Program,
                                           previous_attempt_summary: str,
                                           ancestral_summary: Optional[List[Dict[str, Any]]] = None) -> str: # Method_v2 (added ancestral_summary)
        """Designs a prompt to ask for full code after a diff attempt failed."""
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
    async def get_all_programs(self) -> List[Program]: # Added _v4
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass

    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass


class SelectionControllerInterface(BaseAgent): # Modified_v3
    @abstractmethod
    def select_parents(self, evaluated_programs: List[Program], num_parents: int, task: TaskDefinition) -> List[Program]: # Added task: TaskDefinition _v3
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int, task: TaskDefinition) -> List[Program]: # Added task: TaskDefinition _v3
        pass

    @abstractmethod
    def sort_programs(self, programs: List[Program], task: TaskDefinition) -> List[Program]: # Added _v4
        """Sorts a list of programs according to the agent's selection criteria."""
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass


class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict):
        pass

    @abstractmethod
    async def report_status(self):
        pass

# You can add more specific data classes or interfaces here as needed 