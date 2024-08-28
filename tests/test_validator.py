import os
import pytest
from dotenv import load_dotenv
from guardrails import Guard
from pydantic import BaseModel, Field
from validator.main import RestrictToTopic

load_dotenv()


class ValidatorTestObject(BaseModel):
    test_val: str = Field(
        validators=[
            RestrictToTopic(
                valid_topics=["sports"],
                invalid_topics=["music"],
                disable_classifier=True,
                disable_llm=False,
                on_fail="exception",
            )
        ],
        api_key=os.getenv("OPENAI_API_KEY"),
    )


TEST_OUTPUT = """
{
  "test_val": "In Super Bowl LVII in 2023, the Chiefs clashed with the Philadelphia Eagles in a fiercely contested battle, ultimately emerging victorious with a score of 38-35."
}
"""

def test_validator_pass():
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)
    resp = guard.parse(TEST_OUTPUT)
    assert resp.validation_passed is True

TEST_FAIL_OUTPUT = """
{
"test_val": "The Beatles were a charismatic English pop-rock band of the 1960s."
}
"""

def test_validator_fail():
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)
    with pytest.raises(Exception) as e:
        guard.parse(TEST_FAIL_OUTPUT)
    assert str(e.value) == "Validation failed for field with errors: Invalid topics found: ['music']"