"""Calculator tool.

Safely evaluates basic arithmetic expressions without using eval().
The text is parsed into a syntax tree with Python's ast module, and only
number literals and whitelisted arithmetic operators are evaluated.
Anything else (names, function calls, attribute access, strings, ...) is rejected.
"""

import ast
import logging
import operator
from collections.abc import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Number = int | float

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 10**100
ALLOWED_SYNTAX = "numbers, parentheses and + - * / // % **"

BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when an expression cannot be evaluated safely."""


class CalculatorInput(BaseModel):
    """Arguments the LLM must provide when calling the calculator tool."""

    expression: str = Field(
        description="Arithmetic expression to evaluate, for example '(25 * 4) / 2'."
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Supports numbers, parentheses and the operators + - * / // % **.
    Use ** 0.5 for square roots. Variables and functions are not supported.
    """
    try:
        result = evaluate_expression(expression)
    except CalculatorError as exc:
        logger.info("Calculator rejected %r: %s", expression, exc)
        return f"Error: {exc}"
    return format_number(result)


def evaluate_expression(expression: str) -> Number:
    """Safely evaluate an arithmetic expression.

    Raises CalculatorError if the expression is empty, malformed, uses
    unsupported syntax, divides by zero, or produces a number that is too large.
    """
    expression = expression.strip()
    if not expression:
        raise CalculatorError("expression is empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"expression is too long (maximum is {MAX_EXPRESSION_LENGTH} characters)"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        raise CalculatorError(f"invalid expression: {expression}") from None

    try:
        return _evaluate_node(tree.body)
    except ZeroDivisionError:
        raise CalculatorError("division by zero") from None
    except OverflowError:
        raise CalculatorError("number is too large") from None


def format_number(value: Number) -> str:
    """Format a result for display: 25.0 -> '25', 1/3 -> '0.333333333333'."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _evaluate_node(node: ast.expr) -> Number:
    """Recursively evaluate one node of the syntax tree."""
    if isinstance(node, ast.Constant):
        value = _evaluate_constant(node)
    elif isinstance(node, ast.UnaryOp):
        value = _evaluate_unary(node)
    elif isinstance(node, ast.BinOp):
        value = _evaluate_binary(node)
    else:
        raise _unsupported(node)
    return _check_result(value)


def _evaluate_constant(node: ast.Constant) -> Number:
    value = node.value
    # bool is a subclass of int in Python, so True/False must be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _unsupported(node)
    return value


def _evaluate_unary(node: ast.UnaryOp) -> Number:
    operation = UNARY_OPERATORS.get(type(node.op))
    if operation is None:
        raise _unsupported(node)
    return operation(_evaluate_node(node.operand))


def _evaluate_binary(node: ast.BinOp) -> Number:
    operation = BINARY_OPERATORS.get(type(node.op))
    if operation is None:
        raise _unsupported(node)
    left = _evaluate_node(node.left)
    right = _evaluate_node(node.right)
    # Without this limit, an input such as 9 ** 9 ** 9 would freeze the program.
    if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
        raise CalculatorError(f"exponent is too large (maximum is {MAX_EXPONENT})")
    return operation(left, right)


def _check_result(value: Number | complex) -> Number:
    # A negative number raised to a fractional power, e.g. (-8) ** (1/3), is complex.
    if isinstance(value, complex):
        raise CalculatorError("result is not a real number")
    if abs(value) > MAX_MAGNITUDE:
        raise CalculatorError("number is too large")
    return value


def _unsupported(node: ast.AST) -> CalculatorError:
    return CalculatorError(
        f"unsupported element: {ast.unparse(node)} (only {ALLOWED_SYNTAX} are allowed)"
    )
