"""
prompt_lint: static lint rules for LLM prompt quality.

Catch injection risks, structure problems, and common mistakes
before the prompt hits the model. Zero runtime deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Violation",
    "LintResult",
    "Rule",
    "TooLong",
    "InjectionRisk",
    "EmptyContent",
    "DuplicateSystemPrompt",
    "MissingSystemPrompt",
    "UnknownRole",
    "PromptLinter",
]

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    rule_name: str
    severity: Literal["error", "warning", "info"]
    message: str
    message_index: int | None = None  # which message triggered it; None = whole-prompt


@dataclass
class LintResult:
    passed: bool  # True when no "error" severity violations
    violations: list[Violation] = field(default_factory=list)


class Rule:
    """Base class for lint rules. Subclasses must set *name* and optionally *severity*."""

    name: str
    severity: Literal["error", "warning", "info"] = "warning"

    def check(self, messages: list[dict]) -> list[Violation]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


class TooLong(Rule):
    """Warn when total character count across all message content exceeds a threshold."""

    name = "too_long"
    severity = "warning"

    def __init__(self, max_chars: int = 50_000) -> None:
        self.max_chars = max_chars

    def check(self, messages: list[dict]) -> list[Violation]:
        total = sum(
            len(m.get("content") or "") for m in messages if isinstance(m.get("content"), str)
        )
        if total > self.max_chars:
            return [
                Violation(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Total prompt length {total} chars exceeds max {self.max_chars}.",
                )
            ]
        return []


# Patterns that suggest prompt injection attempts (case-insensitive).
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore all previous",
        r"ignore previous",
        r"forget everything",
        r"disregard",
        r"you are now",
        r"new instructions:",
    ]
]


class InjectionRisk(Rule):
    """Flag content strings that contain known prompt-injection trigger phrases."""

    name = "injection_risk"
    severity = "error"

    def check(self, messages: list[dict]) -> list[Violation]:
        violations: list[Violation] = []
        for idx, msg in enumerate(messages):
            content = msg.get("content") or ""
            if not isinstance(content, str):
                continue
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            severity=self.severity,
                            message=(
                                f"Message {idx} contains a potential injection phrase "
                                f"matching '{pattern.pattern}'."
                            ),
                            message_index=idx,
                        )
                    )
                    # one violation per message is enough; move to next message
                    break
        return violations


class EmptyContent(Rule):
    """Warn when a message has empty, None, or whitespace-only content."""

    name = "empty_content"
    severity = "warning"

    def check(self, messages: list[dict]) -> list[Violation]:
        violations: list[Violation] = []
        for idx, msg in enumerate(messages):
            content = msg.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Message {idx} has empty or whitespace-only content.",
                        message_index=idx,
                    )
                )
        return violations


class DuplicateSystemPrompt(Rule):
    """Error when more than one message carries role='system'."""

    name = "duplicate_system_prompt"
    severity = "error"

    def check(self, messages: list[dict]) -> list[Violation]:
        system_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if len(system_indices) <= 1:
            return []
        # Report all duplicates beyond the first
        return [
            Violation(
                rule_name=self.name,
                severity=self.severity,
                message=(
                    f"Message {idx} is a duplicate system prompt "
                    f"(first system prompt is at index {system_indices[0]})."
                ),
                message_index=idx,
            )
            for idx in system_indices[1:]
        ]


class MissingSystemPrompt(Rule):
    """Info when no message has role='system'."""

    name = "missing_system_prompt"
    severity = "info"

    def check(self, messages: list[dict]) -> list[Violation]:
        if any(m.get("role") == "system" for m in messages):
            return []
        return [
            Violation(
                rule_name=self.name,
                severity=self.severity,
                message="No system prompt found in messages.",
            )
        ]


_KNOWN_ROLES = {"system", "user", "assistant", "tool"}


class UnknownRole(Rule):
    """Warn when a message role is not in the standard set."""

    name = "unknown_role"
    severity = "warning"

    def check(self, messages: list[dict]) -> list[Violation]:
        violations: list[Violation] = []
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role not in _KNOWN_ROLES:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Message {idx} has unknown role '{role}'.",
                        message_index=idx,
                    )
                )
        return violations


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------


class PromptLinter:
    """Run a set of rules against a prompt (list of message dicts)."""

    DEFAULT_RULES: list[Rule] = [
        TooLong(),
        InjectionRisk(),
        EmptyContent(),
        DuplicateSystemPrompt(),
        MissingSystemPrompt(),
        UnknownRole(),
    ]

    def __init__(self, rules: list[Rule] | None = None) -> None:
        # Use a copy so mutations don't affect the class-level default.
        self._rules: list[Rule] = list(rules) if rules is not None else list(self.DEFAULT_RULES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lint(self, messages: list[dict]) -> LintResult:
        """Run all rules against *messages* and return a LintResult."""
        all_violations: list[Violation] = []
        for rule in self._rules:
            all_violations.extend(rule.check(messages))
        passed = not any(v.severity == "error" for v in all_violations)
        return LintResult(passed=passed, violations=all_violations)

    def lint_text(self, text: str) -> LintResult:
        """Convenience: lint a single raw string as a user message."""
        return self.lint([{"role": "user", "content": text}])

    def add_rule(self, rule: Rule) -> None:
        """Append a rule to the active rule list."""
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        """Remove all rules whose *name* matches *rule_name*."""
        self._rules = [r for r in self._rules if r.name != rule_name]

    def rules(self) -> list[Rule]:
        """Return a copy of the current rule list."""
        return list(self._rules)
