"""Deterministic calculator plus a deliberately bounded, inspectable router."""
from __future__ import annotations
import ast
import operator as op
import re
from dataclasses import dataclass, asdict

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


@dataclass
class Calculation:
    expression: str
    result: float | None
    route: str
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic containing numbers, parentheses, + - * / ** and %."""
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic expressions are allowed.")

    result = visit(tree)
    if abs(result) > 1e15:
        raise ValueError("Result is outside the permitted range.")
    return float(result)


def _number(s: str) -> float:
    return float(s.replace(",", ""))


def route_calculation(query: str) -> Calculation | None:
    """Route only a few transparent calculation forms.

    Supported automatically:
      * explicit `CALC: expression`
      * `X% of Y`
      * `$X per month for Y months/years`

    This is intentionally not a general natural-language math agent. Students can inspect
    and modify these routing rules as a bounded intervention.
    """
    explicit = re.search(r"CALC:\s*(.+)$", query, flags=re.IGNORECASE)
    if explicit:
        expr = explicit.group(1).strip()
        try:
            return Calculation(expr, safe_eval(expr), "explicit_CALC")
        except Exception as e:
            return Calculation(expr, None, "explicit_CALC", str(e))

    pct = re.search(
        r"(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+\$?(?P<amt>\d[\d,]*(?:\.\d+)?)",
        query,
        flags=re.IGNORECASE,
    )
    if pct:
        expr = f"{_number(pct.group('pct'))}/100*{_number(pct.group('amt'))}"
        try:
            return Calculation(expr, safe_eval(expr), "percent_of")
        except Exception as e:
            return Calculation(expr, None, "percent_of", str(e))

    periodic = re.search(
        r"\$?(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?:/|per\s+)month\s+(?:for|over)\s+(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>months?|years?)",
        query,
        flags=re.IGNORECASE,
    )
    if periodic:
        amount = _number(periodic.group("amt"))
        n = _number(periodic.group("n"))
        multiplier = 12 if periodic.group("unit").lower().startswith("year") else 1
        expr = f"{amount}*{n}*{multiplier}"
        try:
            return Calculation(expr, safe_eval(expr), "periodic_total")
        except Exception as e:
            return Calculation(expr, None, "periodic_total", str(e))

    return None


def format_calculator_block(calc: Calculation | None) -> str:
    if calc is None:
        return ""
    if calc.error:
        return (
            "DETERMINISTIC CALCULATOR ERROR\n"
            f"Route: {calc.route}\nExpression: {calc.expression}\nError: {calc.error}"
        )
    return (
        "DETERMINISTIC CALCULATOR\n"
        f"Route: {calc.route}\nExpression: {calc.expression}\nResult: {calc.result:g}"
    )
