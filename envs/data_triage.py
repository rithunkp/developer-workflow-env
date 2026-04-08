from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from pydantic import BaseModel, ValidationError

from envs.base import BaseEnv, StepResult
from graders.data_triage_grader import grade_data_triage


TASK_FILE = Path(__file__).resolve().parent.parent / "tasks" / "data_triage_tasks.json"


class DataTriageObs(BaseModel):
    csv_content: str
    column_names: list[str]
    step: int


class DataTriageAction(BaseModel):
    nulls: list[list[int]]
    duplicates: list[int]


class DataTriageEnv(BaseEnv):
    max_reward = 0.84

    def __init__(self) -> None:
        self._config = json.loads(TASK_FILE.read_text(encoding="utf-8-sig"))
        self._task_name = "data-triage-easy"
        self._seed = 0
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._observation: DataTriageObs | None = None
        self._truth: dict = {"nulls": [], "duplicates": []}

    def _generate_rows(self, seed: int) -> tuple[list[str], list[list[str]], dict]:
        rng = random.Random(seed)
        columns = list(self._config["columns"])
        pools = self._config["pools"]

        rows: list[list[str]] = []
        for row_idx in range(18):
            first = rng.choice(pools["first_names"])
            last = rng.choice(pools["last_names"])
            city = rng.choice(pools["cities"])
            plan = rng.choice(pools["plans"])
            domain = rng.choice(pools["domains"])
            rows.append(
                [
                    f"RID-{seed % 1000:03d}-{row_idx:02d}",
                    f"{first} {last}",
                    f"{first.lower()}.{last.lower()}{row_idx}@{domain}",
                    city,
                    plan,
                    str(rng.randint(1, 540)),
                ]
            )

        null_row_indices = rng.sample(range(len(rows)), 2)
        null_col_indices = rng.sample([1, 2, 3, 4, 5], 2)
        for row_idx, col_idx in zip(null_row_indices, null_col_indices):
            rows[row_idx][col_idx] = ""

        duplicate_sources = rng.sample(range(len(rows)), 2)
        rows.extend([rows[source][:] for source in duplicate_sources])

        final_rows = rows[:]
        rng.shuffle(final_rows)

        seen: dict[tuple[str, ...], int] = {}
        duplicate_row_indices: list[int] = []
        null_positions: list[list[int]] = []
        for row_idx, row in enumerate(final_rows):
            row_key = tuple(row)
            if row_key in seen:
                duplicate_row_indices.append(row_idx)
            else:
                seen[row_key] = row_idx
            for col_idx, value in enumerate(row):
                if value == "":
                    null_positions.append([row_idx, col_idx])

        return columns, final_rows, {"nulls": null_positions, "duplicates": duplicate_row_indices}

    @staticmethod
    def _rows_to_csv(columns: list[str], rows: list[list[str]]) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerows(rows)
        return buffer.getvalue().strip()

    def _build_observation(self) -> DataTriageObs:
        if self._observation is None:
            raise RuntimeError("Environment has not been reset")
        return self._observation

    def reset(self, task: str, seed: int | None = None) -> DataTriageObs:
        self._task_name = task
        default_seed = self._config["default_seeds"][0]
        self._seed = int(default_seed if seed is None else seed)
        self._step = 0
        self._done = False
        self._total_reward = 0.0

        columns, rows, truth = self._generate_rows(self._seed)
        self._truth = truth
        self._observation = DataTriageObs(
            csv_content=self._rows_to_csv(columns, rows),
            column_names=columns,
            step=self._step,
        )
        return self._build_observation()

    def step(self, action: dict) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"score": grade_data_triage({}, self._truth)["score"], "error": "Episode already completed", "step": self._step},
            )

        try:
            parsed_action = DataTriageAction.model_validate(action)
            grade = grade_data_triage(parsed_action.model_dump(), self._truth)
            reward = max(
                0.0,
                (grade["correct_nulls"] * 0.17)
                + (grade["correct_duplicates"] * 0.25)
                - (grade["false_positives"] * 0.10),
            )
            error = None
        except ValidationError as exc:
            grade = grade_data_triage({}, self._truth)
            reward = 0.0
            error = exc.errors()[0]["msg"]

        self._step = 1
        self._done = True
        self._total_reward += reward
        self._observation = self._observation.model_copy(update={"step": self._step})

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=True,
            info={"score": grade["score"], "error": error, "step": self._step},
        )

    def state(self) -> dict:
        observation = self._build_observation().model_dump() if self._observation else {}
        return {
            "observation": observation,
            "step": self._step,
            "done": self._done,
            "total_reward": self._total_reward,
        }
