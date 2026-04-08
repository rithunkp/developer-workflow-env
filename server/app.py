from __future__ import annotations

import random
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from envs.base import BaseEnv
from envs.code_review import CodeReviewEnv
from envs.data_triage import DataTriageEnv
from envs.email_triage import EmailTriageEnv


app = FastAPI(title="OpenEnv Developer Workflow Environment")

TASKS: dict[str, type[BaseEnv]] = {
    "data-triage-easy": DataTriageEnv,
    "email-triage-medium": EmailTriageEnv,
    "code-review-hard": CodeReviewEnv,
}

CURRENT_ENV: BaseEnv | None = None
CURRENT_TASK = ""
CURRENT_SEED = 0
ERROR_STATE: dict[str, Any] | None = None


def _serialize(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload


def _error_payload(message: str, task: str, seed: int) -> JSONResponse:
    global CURRENT_ENV, CURRENT_TASK, CURRENT_SEED, ERROR_STATE
    CURRENT_ENV = None
    CURRENT_TASK = task
    CURRENT_SEED = seed
    ERROR_STATE = {
        "observation": {"error": message, "available_tasks": list(TASKS)},
        "step": 0,
        "done": True,
        "total_reward": 0.0,
    }
    return JSONResponse(content={"observation": ERROR_STATE["observation"], "task": task, "seed": seed}, status_code=200)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    task = body.get("task", "")
    raw_seed = body.get("seed")
    seed = raw_seed if isinstance(raw_seed, int) else random.randint(0, 10**6)

    if not isinstance(task, str) or task not in TASKS:
        return _error_payload(f"Unknown task '{task}'. Use one of {list(TASKS)}", str(task), int(seed))

    global CURRENT_ENV, CURRENT_TASK, CURRENT_SEED, ERROR_STATE
    CURRENT_ENV = TASKS[task]()
    CURRENT_TASK = task
    CURRENT_SEED = int(seed)
    ERROR_STATE = None
    observation = CURRENT_ENV.reset(task=task, seed=int(seed))
    return JSONResponse(content={"observation": _serialize(observation), "task": task, "seed": int(seed)}, status_code=200)


@app.post("/step")
async def step(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    action = body.get("action")
    if CURRENT_ENV is None:
        payload = {
            "observation": (ERROR_STATE or {"observation": {"error": "Call /reset before /step"}})["observation"],
            "reward": 0.0,
            "done": True,
            "info": {"score": 0.0, "error": "Call /reset before /step", "step": 0},
        }
        return JSONResponse(content=payload, status_code=200)

    result = CURRENT_ENV.step(action)
    return JSONResponse(
        content={
            "observation": _serialize(result.observation),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        },
        status_code=200,
    )


@app.get("/state")
async def state() -> JSONResponse:
    if CURRENT_ENV is None:
        return JSONResponse(content=ERROR_STATE or {"observation": {}, "step": 0, "done": True, "total_reward": 0.0}, status_code=200)
    return JSONResponse(content=CURRENT_ENV.state(), status_code=200)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
