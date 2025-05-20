# tests/integration/test_initial_code_flow.py
import pytest
from unittest.mock import AsyncMock  # For mocking async methods!

from core.interfaces import TaskDefinition
from engine.prompting import PromptDesignerAgent
from engine.generation import CodeGeneratorAgent
from config import settings  # To potentially mock settings values


# A fixture for our TaskDefinition
@pytest.fixture
def simple_task_def_fixture_v1_0_0() -> TaskDefinition:
    """A simple TaskDefinition for testing."""
    return TaskDefinition(
        id="test_simple_add",
        description="Create a Python function that adds two numbers.",
        function_name_to_evolve="add_numbers",
        input_output_examples=[{"input": [1, 2], "output": 3}],
        allowed_imports=["math"]  # Just an example
    )


# A fixture for PromptDesignerAgent
@pytest.fixture
def prompt_designer_fixture_v1_0_0(simple_task_def_fixture_v1_0_0: TaskDefinition) -> PromptDesignerAgent:
    """Provides a PromptDesignerAgent instance."""
    return PromptDesignerAgent(task_definition=simple_task_def_fixture_v1_0_0)


# A fixture for CodeGeneratorAgent (can reuse from unit tests if structured well)
@pytest.fixture
def code_gen_agent_fixture_v1_0_0(mocker) -> CodeGeneratorAgent:  # mocker is from pytest-mock
    """Provides a CodeGeneratorAgent instance with GEMINI_API_KEY mocked."""
    mocker.patch.object(settings, 'GEMINI_API_KEY', "DUMMY_KEY_FOR_TESTING_INTEGRATION")
    # If other settings are crucial for CodeGeneratorAgent init, mock them too
    # mocker.patch.object(settings, 'GEMINI_PRO_MODEL_NAME', "dummy-model-integration")
    return CodeGeneratorAgent()


@pytest.mark.asyncio  # Pytest needs this for async test functions!
async def test_initial_prompt_to_code_generation_v1_0_0(
        prompt_designer_fixture_v1_0_0: PromptDesignerAgent,
        code_gen_agent_fixture_v1_0_0: CodeGeneratorAgent,
        simple_task_def_fixture_v1_0_0: TaskDefinition,
        mocker  # For mocking the LLM call within CodeGeneratorAgent
):
    """
    Test that an initial prompt leads to code generation,
    mocking the actual LLM call.
    """
    # 1. Arrange: Design the initial prompt
    initial_prompt = prompt_designer_fixture_v1_0_0.design_initial_prompt(simple_task_def_fixture_v1_0_0)
    assert "add_numbers" in initial_prompt  # Quick check the prompt is reasonable
    assert "adds two numbers" in initial_prompt

    # 2. Arrange: Mock the LLM's response from CodeGeneratorAgent's generate_code method
    # This is the magic part! We're telling generate_code what to return without calling Gemini!
    mocked_llm_response_code = "def add_numbers(a, b):\n  return a + b"

    # We need to mock the internal `model_to_use.generate_content_async` if that's what's called,
    # or the public `generate_code` method of the `code_gen_agent_fixture_v1_0_0` instance.
    # Let's mock the higher-level `generate_code` for simplicity here, or rather its API call.
    # The actual method called by `code_gen_agent_fixture_v1_0_0.execute` is `generate_code`
    # which itself calls `model_to_use.generate_content_async`.
    # So, we find the `generate_content_async` method of the model object it creates.

    # Simpler: Let's mock the generate_code method of the instance itself for this integration test's purpose.
    # If generate_code calls other internal methods we want to test (like _clean_llm_output),
    # we should mock the *actual API call inside* generate_code.

    # Let's assume generate_code calls an internal _call_llm_api_async method (hypothetical)
    # For your actual code, it's `model_to_use.generate_content_async`
    # We'll use mocker.patch.object for the instance's method or a deeper path.

    # To mock the `generate_content_async` call inside `code_gen_agent_fixture_v1_0_0.generate_code`:
    # We'd need to know how `model_to_use` is obtained. Since it's created inside,
    # it's easier to mock `CodeGeneratorAgent.generate_code` itself if we only care about its final output for this integration test.
    # Or, if `execute` calls `generate_code`, we mock `generate_code`.

    # Let's mock `code_gen_agent_fixture_v1_0_0.generate_code` directly for this example
    # because `execute` calls it.
    mock_generate_code_method = AsyncMock(return_value=mocked_llm_response_code)
    mocker.patch.object(code_gen_agent_fixture_v1_0_0, 'generate_code', new=mock_generate_code_method)

    # 3. Act: Call the CodeGeneratorAgent's execute method
    # (which internally would call the now-mocked generate_code)
    generated_output = await code_gen_agent_fixture_v1_0_0.execute(
        prompt=initial_prompt,
        output_format="code"  # We want full code for initial generation
    )

    # 4. Assert
    # Check if our mock was called
    mock_generate_code_method.assert_called_once_with(
        prompt=initial_prompt,
        model_name=None,  # Or whatever default it uses
        temperature=None,  # Or whatever default it uses
        output_format="code"
    )

    # Check if the output is what our mock was supposed to return
    assert generated_output == mocked_llm_response_code
    assert "def add_numbers(a, b):" in generated_output  # Check content