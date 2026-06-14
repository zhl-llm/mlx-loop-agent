from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RunnableLike(Protocol):
    """Minimal protocol compatible with LangChain Runnable.invoke."""

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LoopRunResult:
    """Result of a loop-agent run."""

    state: dict[str, Any]
    steps: list[dict[str, Any]]
    iterations: int
    finished: bool


class LangChainLoopAgent:
    """A small loop agent that repeatedly invokes a LangChain-compatible runnable."""

    def __init__(self, runnable: RunnableLike, max_iterations: int = 10) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._runnable = runnable
        self._max_iterations = max_iterations

    def run(self, initial_state: dict[str, Any] | None = None) -> LoopRunResult:
        state = dict(initial_state or {})
        steps: list[dict[str, Any]] = []

        for idx in range(self._max_iterations):
            response = self._runnable.invoke(state)
            if not isinstance(response, dict):
                raise TypeError("runnable.invoke must return a dict")

            steps.append(response)
            state.update(response.get("state", {}))

            if response.get("done", False):
                return LoopRunResult(
                    state=state,
                    steps=steps,
                    iterations=idx + 1,
                    finished=True,
                )

        return LoopRunResult(
            state=state,
            steps=steps,
            iterations=self._max_iterations,
            finished=False,
        )
