from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ValidationError

from envs.base import BaseEnv, StepResult
from graders.email_triage_grader import grade_email_triage


TASK_FILE = Path(__file__).resolve().parent.parent / "tasks" / "email_triage_tasks.json"
PHASES = ["classify", "prioritize", "route"]


class EmailTriageObs(BaseModel):
    emails: list[dict]
    current_step_type: str
    step: int


class EmailTriageAction(BaseModel):
    email_id: str
    category: str
    priority: str
    note: str | None = None


class EmailTriageEnv(BaseEnv):
    max_steps = 30
    max_reward = 0.65

    def __init__(self) -> None:
        self._scenarios = json.loads(TASK_FILE.read_text(encoding="utf-8-sig"))
        self._task_name = "email-triage-medium"
        self._seed = 0
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._phase_index = 0
        self._cursor = 0
        self._emails: list[dict] = []
        self._answers: dict[str, dict] = {}
        self._submissions: dict[str, dict] = {}
        self._rewarded_pairs: set[str] = set()
        self._rewarded_notes: set[str] = set()

    def _build_scenario(self, seed: int) -> list[dict]:
        rng = random.Random(seed)
        scenario = self._scenarios[seed % len(self._scenarios)]
        emails = [dict(email) for email in scenario["emails"]]
        rng.shuffle(emails)
        return emails

    def _current_phase(self) -> str:
        return PHASES[self._phase_index]

    def _current_email(self) -> dict:
        return self._emails[self._cursor]

    def _terminal_observation(self) -> EmailTriageObs:
        return EmailTriageObs(emails=[], current_step_type="route", step=self._step)

    def _build_observation(self) -> EmailTriageObs:
        if self._done or self._cursor >= len(self._emails):
            return self._terminal_observation()
        current_email = self._current_email()
        return EmailTriageObs(
            emails=[{"id": current_email["id"], "subject": current_email["subject"], "body": current_email["body"]}],
            current_step_type=self._current_phase(),
            step=self._step,
        )

    def _advance(self) -> None:
        self._phase_index += 1
        if self._phase_index >= len(PHASES):
            self._phase_index = 0
            self._cursor += 1
        if self._cursor >= len(self._emails) or self._step >= self.max_steps:
            self._done = True

    def reset(self, task: str, seed: int | None = None) -> EmailTriageObs:
        self._task_name = task
        self._seed = int(0 if seed is None else seed)
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._phase_index = 0
        self._cursor = 0
        self._submissions = {}
        self._rewarded_pairs = set()
        self._rewarded_notes = set()
        self._emails = self._build_scenario(self._seed)
        self._answers = {email["id"]: email["answer"] for email in self._emails}
        return self._build_observation()

    def step(self, action: dict) -> StepResult:
        if self._done:
            grade = grade_email_triage(self._submissions, self._answers)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"score": grade["score"], "error": "Episode already completed", "step": self._step},
            )

        reward = 0.0
        error = None

        try:
            parsed_action = EmailTriageAction.model_validate(action)
        except ValidationError as exc:
            grade = grade_email_triage(self._submissions, self._answers)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=False,
                info={"score": grade["score"], "error": exc.errors()[0]["msg"], "step": self._step},
            )

        current_email = self._current_email()
        if parsed_action.email_id != current_email["id"]:
            grade = grade_email_triage(self._submissions, self._answers)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=False,
                info={"score": grade["score"], "error": f"Expected email_id {current_email['id']} during {self._current_phase()} phase", "step": self._step},
            )

        truth = self._answers[parsed_action.email_id]
        self._submissions[parsed_action.email_id] = parsed_action.model_dump()

        if (
            parsed_action.category == truth["category"]
            and parsed_action.priority == truth["priority"]
            and parsed_action.email_id not in self._rewarded_pairs
        ):
            reward += 0.075
            self._rewarded_pairs.add(parsed_action.email_id)

        if truth["category"] == "spam" and parsed_action.priority == "urgent":
            reward -= 0.05

        if self._current_phase() == "route" and truth["priority"] == "urgent":
            keyword_tokens = {token.lower() for token in truth.get("note_keywords", [])}
            note_tokens = set((parsed_action.note or "").lower().replace("-", " ").split())
            if keyword_tokens & note_tokens and parsed_action.email_id not in self._rewarded_notes:
                reward += 0.025
                self._rewarded_notes.add(parsed_action.email_id)
            elif not parsed_action.note:
                error = "Urgent emails require a routing note during route phase"

        self._step += 1
        self._advance()
        self._total_reward += reward
        grade = grade_email_triage(self._submissions, self._answers)
        observation = self._build_observation().model_dump()

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info={"score": grade["score"], "error": error, "step": self._step},
        )

    def state(self) -> dict:
        return {
            "observation": self._build_observation().model_dump(),
            "step": self._step,
            "done": self._done,
            "total_reward": self._total_reward,
        }
