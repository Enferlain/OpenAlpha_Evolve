# tests/integration/test_initial_code_flow_with_api.py
import pytest

from core.interfaces import TaskDefinition
from engine.prompting import PromptStudio
from engine.generation import CodeProducer
from config import settings


# You'd still use your fixtures for setup!
@pytest.fixture
def simple_task_def_fixture_v1_0_0() -> TaskDefinition:
    return TaskDefinition(
        id="test_api_simple_add",
        description="Create a Python function `add_two(x, y)` that returns the sum of x and y.",
        evolve_function="add_two",
        io_examples=[{"input": [1, 2], "output": 3}],
        allowed_imports=[]
    )


@pytest.fixture
def prompt_designer_fixture_v1_0_0(simple_task_def_fixture_v1_0_0: TaskDefinition) -> PromptStudio:
    return PromptStudio(task_definition=simple_task_def_fixture_v1_0_0)


@pytest.fixture
def real_code_gen_agent_fixture_v1_0_0() -> CodeProducer:
    # IMPORTANT: Ensure GEMINI_API_KEY is *actually* set in your environment or .env file
    # for these tests to work! Pytest will pick it up via python-dotenv.
    if not settings.GEMINI_API_KEY or "NON-FUNCTIONAL" in settings.GEMINI_API_KEY or "NOT_FOUND" in settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not configured for real API tests, skipping.")
    # You might want to explicitly use the Flash model for these tests in settings
    # or override it if the agent allows model selection per call.
    return CodeProducer()  # Assumes settings are configured for Gemini Flash


@pytest.mark.llm_api  # Custom marker!
@pytest.mark.asyncio
async def test_initial_prompt_to_real_code_generation_v1_0_0(
        prompt_designer_fixture_v1_0_0: PromptStudio,
        real_code_gen_agent_fixture_v1_0_0: CodeProducer,  # Using the 'real' agent
        simple_task_def_fixture_v1_0_0: TaskDefinition
):
    """
    Test that an initial prompt leads to code generation using the REAL Gemini API.
    This test is expected to be slower and potentially non-deterministic.
    """
    # 1. Arrange: Design the initial prompt
    initial_prompt = prompt_designer_fixture_v1_0_0.initial_prompt(simple_task_def_fixture_v1_0_0)
    assert "add_two" in initial_prompt

    # 2. Act: Call the CodeProducer's execute method - NO MOCKING HERE!
    # This will make a real API call to Gemini.
    # We might want to add a small delay if we're running many such tests in sequence
    # to be kind to the API, though for a few tests it's usually fine.
    # await asyncio.sleep(1) # Optional: if running many API tests back-to-back

    generated_output_code = ""
    try:
        generated_output_code = await real_code_gen_agent_fixture_v1_0_0.execute(
            prompt=initial_prompt,
            model_name=settings.GEMINI_FLASH_MODEL_NAME,  # Explicitly use Flash
            temperature=0.7,  # Control temperature for some consistency
            output_format="code"
        )
    except Exception as e:
        pytest.fail(f"Real API call for code generation failed: {e}")

    # 3. Assert: Be more flexible with assertions!
    assert generated_output_code is not None, "LLM returned None"
    assert isinstance(generated_output_code, str), "LLM did not return a string"
    assert len(generated_output_code.strip()) > 0, "LLM returned empty code"

    # Check for essential elements rather than exact match
    assert f"def {simple_task_def_fixture_v1_0_0.evolve_function}(" in generated_output_code, "Function definition not found"
    assert "return" in generated_output_code.lower(), "'return' keyword not found"

    # Optional: Try to compile the generated code to see if it's valid Python
    # This is a good check for real LLM output.
    try:
        compile(generated_output_code, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Generated code has a syntax error: {e}\nCode:\n{generated_output_code}")

    print(
        f"\n--- Generated Code by Real API for Task '{simple_task_def_fixture_v1_0_0.id}' ---\n{generated_output_code}\n---")