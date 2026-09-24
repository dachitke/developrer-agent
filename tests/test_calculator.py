"""Tests for the calculator tool."""

import pytest

from app.tools.calculator import CalculatorError, calculator, evaluate_expression


def run_tool(expression: str) -> str:
    """Call the tool the same way the agent will: with a dict of arguments."""
    return calculator.invoke({"expression": expression})


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("25 * 4", "100"),
        ("125 / 5", "25"),
        ("7 / 2", "3.5"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("-5 + 2", "-3"),
        ("2 ** 10", "1024"),
        ("16 ** 0.5", "4"),
        ("17 % 5", "2"),
        ("17 // 5", "3"),
        ("0.1 + 0.2", "0.3"),
        ("1 / 3", "0.333333333333"),
        ("  10 - 4  ", "6"),
    ],
)
def test_valid_expressions(expression: str, expected: str) -> None:
    assert run_tool(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected_error"),
    [
        ("10 / 0", "division by zero"),
        ("10 // 0", "division by zero"),
        ("10 % 0", "division by zero"),
        ("", "expression is empty"),
        ("   ", "expression is empty"),
        ("2 +", "invalid expression"),
        ("(2 + 3", "invalid expression"),
        ("2 ^ 3", "unsupported element"),
        ("1 < 2", "unsupported element"),
        ("True + 1", "unsupported element"),
        ("'a' * 3", "unsupported element"),
        ("2j + 1", "unsupported element"),
        ("x + 1", "unsupported element"),
        ("sqrt(16)", "unsupported element"),
        ("(-8) ** (1 / 3)", "result is not a real number"),
        ("1" * 201, "expression is too long"),
    ],
)
def test_invalid_expressions_return_error_instead_of_raising(
    expression: str, expected_error: str
) -> None:
    result = run_tool(expression)

    assert result.startswith("Error:")
    assert expected_error in result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "open('secret.txt').read()",
        "(1).__class__.__bases__",
        "[x for x in range(10)]",
        "lambda: 1",
    ],
)
def test_code_injection_is_rejected(expression: str) -> None:
    assert run_tool(expression).startswith("Error: unsupported element")


@pytest.mark.parametrize(
    ("expression", "expected_error"),
    [
        ("9 ** 9 ** 9", "exponent is too large"),
        ("2 ** 10000", "exponent is too large"),
        ("10 ** 100 * 10", "number is too large"),
        ("1e400", "number is too large"),
    ],
)
def test_huge_computations_are_rejected(expression: str, expected_error: str) -> None:
    result = run_tool(expression)

    assert result.startswith("Error:")
    assert expected_error in result


def test_evaluate_expression_returns_number_or_raises_calculator_error() -> None:
    assert evaluate_expression("6 * 7") == 42

    with pytest.raises(CalculatorError, match="division by zero"):
        evaluate_expression("1 / 0")


def test_tool_exposes_name_description_and_argument_to_the_llm() -> None:
    assert calculator.name == "calculator"
    assert "arithmetic expression" in calculator.description
    assert list(calculator.args) == ["expression"]
