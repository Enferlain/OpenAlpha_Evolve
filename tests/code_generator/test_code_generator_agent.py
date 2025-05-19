import pytest  # We need to import pytest to use its features, though not always for basic tests
from code_generator.code_generator_agent import CodeGeneratorAgent


# We can instantiate the agent once if its __init__ is simple, or do it per test.
# For _clean_llm_output, it doesn't rely on the agent's state, so we can even call it directly
# if we made it a static method, or just instantiate a dummy agent.
# Let's assume we need an instance for this example.

# You might need to set a dummy GEMINI_API_KEY for the agent to initialize,
# or mock settings.GEMINI_API_KEY if CodeGeneratorAgent's __init__ depends on it.
# For this specific method, we might not even need a full agent if it was static,
# but let's practice with an instance.

@pytest.fixture
def code_gen_agent(mocker):  # Using a fixture for reusable setup!
    """Provides a CodeGeneratorAgent instance for tests."""
    # Mock settings.GEMINI_API_KEY to avoid issues if it's checked in __init__
    mocker.patch('config.settings.GEMINI_API_KEY', "DUMMY_KEY_FOR_TESTING")
    # If there are other essential configs, mock them too.
    # mocker.patch('config.settings.GEMINI_PRO_MODEL_NAME', "dummy-model") # If needed
    return CodeGeneratorAgent()


# Test case 1: Cleaning Python markdown fences
def test_clean_llm_output_with_python_fences_v1_0_0(code_gen_agent: CodeGeneratorAgent):
    """Test _clean_llm_output with ```python ... ``` fences."""  # Docstring explains the test!
    raw_code = "```python\ndef hello():\n  print('Hello, Onii-chan!')\n```"
    expected_clean_code = "def hello():\n  print('Hello, Onii-chan!')"

    actual_clean_code = code_gen_agent._clean_llm_output(raw_code)

    assert actual_clean_code == expected_clean_code  # The magic assertion! ✨


# Test case 2: Cleaning generic markdown fences
def test_clean_llm_output_with_generic_fences_v1_0_0(code_gen_agent: CodeGeneratorAgent):
    """Test _clean_llm_output with ``` ... ``` fences."""
    raw_code = "```\ndef goodbye():\n  print('Bye bye!')\n```"
    expected_clean_code = "def goodbye():\n  print('Bye bye!')"

    actual_clean_code = code_gen_agent._clean_llm_output(raw_code)

    assert actual_clean_code == expected_clean_code


# Test case 3: Already clean code
def test_clean_llm_output_already_clean_v1_0_0(code_gen_agent: CodeGeneratorAgent):
    """Test _clean_llm_output with code that needs no cleaning."""
    raw_code = "def already_clean():\n  pass"
    # The method does a .strip(), so if input is " code ", output is "code"
    expected_clean_code = "def already_clean():\n  pass"

    actual_clean_code = code_gen_agent._clean_llm_output(raw_code)

    assert actual_clean_code == expected_clean_code


# Test case 4: Empty input
def test_clean_llm_output_empty_input_v1_0_0(code_gen_agent: CodeGeneratorAgent):
    """Test _clean_llm_output with an empty string."""
    raw_code = ""
    expected_clean_code = ""

    actual_clean_code = code_gen_agent._clean_llm_output(raw_code)

    assert actual_clean_code == expected_clean_code


# Test case 5: Code with leading/trailing whitespace
def test_clean_llm_output_with_whitespace_v1_0_0(code_gen_agent: CodeGeneratorAgent):
    """Test _clean_llm_output strips external whitespace."""
    raw_code = "  \n```python\nprint('spaced')\n```  \n  "
    expected_clean_code = "print('spaced')"  # .strip() happens after fence removal

    actual_clean_code = code_gen_agent._clean_llm_output(raw_code)

    assert actual_clean_code == expected_clean_code