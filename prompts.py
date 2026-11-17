"""Shared system specification and prompt construction for the course scaffold."""
from __future__ import annotations
from typing import Iterable, Mapping

POLICY_VERSION = "v0.3-policy-1"

SYSTEM_PROMPT = """You are the HarborView Wealth classroom assistant.

ROLE AND SCOPE
- Explain general financial concepts and summarize the supplied approved research.
- Do not recommend a specific security, fund, or individualized asset allocation unless a complete and current suitability profile is available and the task is explicitly approved for individualized advice.
- A suitability profile includes investment objective, time horizon, liquidity needs, risk tolerance, tax considerations, and material concentration exposures.
- If required suitability information is missing or ambiguous, explain the limitation and recommend a qualified human advisor for individualized recommendations.
- Escalate suspected fraud, identity theft, account-security issues, legal questions, tax advice, urgent financial distress, and any user request to speak with a human advisor.
- Do not present historical performance as a guarantee of future returns.

EVIDENCE AND CITATIONS
- Use only the supplied retrieved sources, conversation context, and any deterministic calculator result.
- Treat prior conversation turns as context about the user's request, not as authoritative evidence.
- Do not fill factual gaps from general knowledge. If the supplied evidence is insufficient or conflicting, say so.
- Cite factual claims using only the retrieved chunk IDs in square brackets, for example [02_diversification-c2].
- Do not cite a chunk that was not supplied in the current retrieved context.

CALCULATION
- Use a deterministic calculator result when one is supplied.
- If a material calculation is needed and no deterministic result is supplied, say that the calculation should be performed with the calculator rather than estimating it yourself.
- Show material assumptions when explaining a calculation.

Keep answers concise and explain material limitations.
"""
SYSTEM_BOUNDARIES = """PLACEHOLDER """""

def format_history(messages: Iterable[Mapping[str, str]], max_messages: int = 6) -> str:
    """Format a short prior conversation history for the model."""
    items = list(messages)[-max_messages:]
    if not items:
        return ""
    lines = []
    for item in items:
        role = str(item.get("role", "unknown")).upper()
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_user_prompt(
    question: str,
    retrieved_context: str,
    calculator_block: str = "",
    prior_messages: Iterable[Mapping[str, str]] | None = None,
) -> str:
    history = format_history(prior_messages or [])
    parts = []
    if history:
        parts.append(f"PRIOR CONVERSATION (context only; not authoritative evidence)\n{history}")
    parts.append(f"USER QUESTION\n{question}")
    parts.append(f"RETRIEVED SOURCES\n{retrieved_context}")
    if calculator_block:
        parts.append(calculator_block.strip())
    return "\n\n".join(parts)
