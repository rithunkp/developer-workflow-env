from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class StepResult(BaseModel):
    observation: Any
    reward: float
    done: bool
    info: dict


class BaseEnv:
    def reset(self, task: str, seed: int | None = None) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> StepResult:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError
