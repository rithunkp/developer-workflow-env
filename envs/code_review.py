from __future__ import annotations

import difflib
import json
import random
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from envs.base import BaseEnv, StepResult
from graders.code_review_grader import grade_code_review


TASK_FILE = Path(__file__).resolve().parent.parent / "tasks" / "code_review_tasks.json"


class CodeReviewObs(BaseModel):
    diff: str
    filename: str
    current_phase: str
    issues_found: list[dict]
    step: int


class FindingAction(BaseModel):
    phase: Literal["identify"]
    issue_type: str
    line: int
    description: str


class FixAction(BaseModel):
    phase: Literal["fix"]
    line: int
    original: str
    fixed: str
    rationale: str


class SummaryAction(BaseModel):
    phase: Literal["summarize"]
    approved: bool
    risk_level: str
    summary: str


CodeReviewAction = Annotated[Union[FindingAction, FixAction, SummaryAction], Field(discriminator="phase")]
ACTION_ADAPTER = TypeAdapter(CodeReviewAction)


class CodeReviewEnv(BaseEnv):
    max_steps = 40
    max_reward = 0.749

    def __init__(self) -> None:
        self._config = json.loads(TASK_FILE.read_text(encoding="utf-8-sig"))
        self._task_name = "code-review-hard"
        self._seed = 0
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._filename = ""
        self._diff = ""
        self._truth_issues: list[dict] = []
        self._issues_found: list[dict] = []
        self._fixes: list[dict] = []
        self._summary_action: dict | None = None

    def _current_phase(self) -> str:
        unique_true_findings = {
            (item["issue_type"], item["line"])
            for item in self._issues_found
            if item.get("matched")
        }
        if self._summary_action is not None:
            return "summarize"
        if len(unique_true_findings) < 4 and self._step < 12:
            return "identify"
        if len(self._fixes) < 3 and self._step < 24:
            return "fix"
        return "summarize"

    def _build_case(self, seed: int) -> tuple[str, str, list[dict]]:
        rng = random.Random(seed)
        filename = self._config["filenames"][seed % len(self._config["filenames"])]
        function_name = rng.choice(self._config["function_names"])
        payload_key = rng.choice(self._config["payload_keys"])
        token_default = rng.choice(self._config["token_defaults"])
        log_path = rng.choice(self._config["log_paths"])

        clean_lines = [
            "import json",
            "import subprocess",
            "from pathlib import Path",
            "",
            f"def {function_name}(request, archive_path, user_role, retries, max_retries):",
            '    token = request.headers.get("X-Token")',
            f'    payload = request.get("{payload_key}", "{{}}")',
            "    records = json.loads(payload)",
            "    should_retry = retries < max_retries",
            '    has_access = user_role in {"admin", "maintainer"}',
            "    cleaned_items = []",
            "    for item in records:",
            "        cleaned_items.append(item.strip())",
            f'    Path("{log_path}").write_text("request received")',
            '    command = ["tar", "-xf", archive_path]',
            "    result = subprocess.check_output(command, text=True)",
            '    return {"token": token, "records": cleaned_items, "retry": should_retry, "access": has_access, "result": result}',
        ]

        buggy_lines = [
            "import json",
            "import subprocess",
            "from pathlib import Path",
            "",
            f"def {function_name}(request, archive_path, user_role, retries, max_retries):",
            f'    token = request.args.get("token", "{token_default}")',
            f'    payload = request.get("{payload_key}", "{{}}")',
            "    records = eval(payload)",
            "    should_retry = retries > max_retries",
            '    has_access = user_role == "admin" or "maintainer"',
            "    cleanedItems=[]",
            "    for Item in records:",
            "        cleanedItems.append(Item.strip())",
            f'    Path("{log_path}").write_text("request received")',
            '    command = f"tar -xf {archive_path}"',
            "    result = subprocess.check_output(command, shell=True, text=True)",
            '    return {"token": token, "records": cleanedItems, "retry": should_retry, "access": has_access, "result": result}',
        ]

        diff = "\n".join(
            difflib.unified_diff(
                clean_lines,
                buggy_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )

        truth_issues = [
            {"issue_type": "security", "line": 6, "description": "Reads token from query args with an insecure fallback", "buggy": buggy_lines[5], "token_default": token_default},
            {"issue_type": "security", "line": 8, "description": "Uses eval on request payload", "buggy": buggy_lines[7], "token_default": token_default},
            {"issue_type": "logic", "line": 9, "description": "Retry condition is inverted", "buggy": buggy_lines[8], "token_default": token_default},
            {"issue_type": "logic", "line": 10, "description": "Access check always evaluates truthy", "buggy": buggy_lines[9], "token_default": token_default},
            {"issue_type": "style", "line": 11, "description": "Variable name and spacing violate style conventions", "buggy": buggy_lines[10], "token_default": token_default},
            {"issue_type": "style", "line": 12, "description": "Loop variable uses inconsistent capitalization", "buggy": buggy_lines[11], "token_default": token_default},
            {"issue_type": "security", "line": 16, "description": "Runs subprocess with shell=True and interpolated command", "buggy": buggy_lines[15], "token_default": token_default},
        ]

        return filename, diff, truth_issues

    def _build_observation(self) -> CodeReviewObs:
        return CodeReviewObs(
            diff=self._diff,
            filename=self._filename,
            current_phase=self._current_phase(),
            issues_found=self._issues_found,
            step=self._step,
        )

    def reset(self, task: str, seed: int | None = None) -> CodeReviewObs:
        self._task_name = task
        self._seed = int(self._config["sample_seeds"][0] if seed is None else seed)
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._issues_found = []
        self._fixes = []
        self._summary_action = None
        self._filename, self._diff, self._truth_issues = self._build_case(self._seed)
        return self._build_observation()

    def step(self, action: dict) -> StepResult:
        if self._done:
            grade = grade_code_review(self._issues_found, self._fixes, self._summary_action, self._truth_issues)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"score": grade["score"], "error": "Episode already completed", "step": self._step},
            )

        reward = 0.0
        error = None

        try:
            parsed_action = ACTION_ADAPTER.validate_python(action)
        except ValidationError as exc:
            grade = grade_code_review(self._issues_found, self._fixes, self._summary_action, self._truth_issues)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=False,
                info={"score": grade["score"], "error": exc.errors()[0]["msg"], "step": self._step},
            )

        truth_lookup = {(issue["issue_type"], issue["line"]): issue for issue in self._truth_issues}
        identified_truth_keys = {
            (item["issue_type"], item["line"])
            for item in self._issues_found
            if item.get("matched")
        }

        if isinstance(parsed_action, FindingAction):
            key = (parsed_action.issue_type.lower(), parsed_action.line)
            matched = key in truth_lookup and key not in identified_truth_keys
            self._issues_found.append(
                {
                    "issue_type": parsed_action.issue_type.lower(),
                    "line": parsed_action.line,
                    "description": parsed_action.description,
                    "matched": matched,
                }
            )
            if matched:
                reward += 0.057
            else:
                reward -= 0.040
        elif isinstance(parsed_action, FixAction):
            self._fixes.append(parsed_action.model_dump())
            issue = next((item for item in self._truth_issues if item["line"] == parsed_action.line), None)
            if issue and (issue["issue_type"], issue["line"]) in identified_truth_keys:
                grade_preview = grade_code_review(self._issues_found, [parsed_action.model_dump()], None, [issue])
                if grade_preview["fix_scores"] and max(grade_preview["fix_scores"]) >= 1.0:
                    reward += 0.050
        else:
            self._summary_action = parsed_action.model_dump()
            unresolved_security = [
                issue
                for issue in self._truth_issues
                if issue["issue_type"] == "security" and (issue["issue_type"], issue["line"]) not in identified_truth_keys
            ]
            if parsed_action.approved and unresolved_security:
                reward -= 0.100
            self._done = True

        self._step += 1
        if self._step >= self.max_steps:
            self._done = True

        self._total_reward += reward
        grade = grade_code_review(self._issues_found, self._fixes, self._summary_action, self._truth_issues)

        return StepResult(
            observation=self._build_observation(),
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
