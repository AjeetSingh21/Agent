"""Tool-level tests. No network, no API key required."""

import pytest

from src.agent.tools import calculator


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", "4"),
        ("2 + 2 * 10", "22"),
        ("(1500*0.07)/12", "8.75"),
        ("2**10", "1024"),
        ("-5 + 3", "-2"),
        ("7 % 3", "1"),
        ("10 // 3", "3"),
    ],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    result = calculator(expression)
    assert result.ok, result.error
    assert result.value == expected


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo pwned")',
        "open('/etc/passwd').read()",
        "[].__class__.__base__",
        "lambda: 1",
        "1/0",
        "9**9**9",
        "'a'*100",
        "not_a_number",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expression):
    result = calculator(expression)
    assert not result.ok
    assert result.error


def test_calculator_records_latency():
    assert calculator("1+1").latency_s >= 0
