---
title: Developer Workflow Env
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# OpenEnv Developer Workflow Environment

## 1. Environment overview and motivation

This project implements a complete OpenEnv-compatible environment server for developer workflow automation. It covers three realistic task families that show up in support and engineering operations: spotting bad data in CSV files, triaging customer email, and reviewing Python pull requests for security, logic, and style defects. The goal is to give an autonomous agent a small but realistic benchmark where actions can be evaluated deterministically through `reset()`, `step()`, and `state()` endpoints.

## 2. Action space definition (per task)

### Data triage (`data-triage-easy`)

- `nulls: list[list[int]]` identifies empty cells as `[row_idx, col_idx]`.
- `duplicates: list[int]` identifies row indices that duplicate an earlier row.

### Email triage (`email-triage-medium`)

- `email_id: str` selects the email currently under review.
- `category: str` must be one of `billing`, `technical`, `general`, or `spam`.
- `priority: str` must be one of `urgent`, `normal`, or `low`.
- `note: str | None` is optional except during urgent routing work, where a useful routing note is expected.

### Code review (`code-review-hard`)

- `FindingAction`
  - `phase: "identify"`
  - `issue_type: "security" | "logic" | "style"`
  - `line: int`
  - `description: str`
- `FixAction`
  - `phase: "fix"`
  - `line: int`
  - `original: str`
  - `fixed: str`
  - `rationale: str`
- `SummaryAction`
  - `phase: "summarize"`
  - `approved: bool`
  - `risk_level: "high" | "medium" | "low"`
  - `summary: str`

## 3. Observation space definition (per task)

### Data triage

- `csv_content: str` raw CSV text containing 20 data rows.
- `column_names: list[str]`
- `step: int`

### Email triage

- `emails: list[dict]` containing the current email payload as `id`, `subject`, and `body`.
- `current_step_type: "classify" | "prioritize" | "route"`
- `step: int`

### Code review

- `diff: str` unified diff of the Python file under review.
- `filename: str`
- `current_phase: "identify" | "fix" | "summarize"`
- `issues_found: list[dict]` containing accumulated findings and whether they matched a true issue.
- `step: int`

## 4. Task descriptions with difficulty levels

- `data-triage-easy`: single-turn CSV inspection with deterministic null and duplicate labels.
- `email-triage-medium`: multi-step support operations workflow over seeded email scenarios.
- `code-review-hard`: iterative PR review over a seeded Python diff with seven planted issues.

## 5. Reward function explanation

### Data triage

- `+0.17` per correct null cell
- `+0.25` per correct duplicate row
- `-0.10` per false positive
- Final grader score: normalized null accuracy, duplicate accuracy, and false-positive penalty clamped into the open interval `(0, 1)` for validator compatibility

### Email triage

- `+0.075` the first time an email receives the correct `(category, priority)` pair
- `+0.025` for an urgent-email routing note that includes a correct issue keyword
- `-0.05` whenever spam is marked as urgent
- Final grader score: weighted category accuracy, priority accuracy, and urgent-note keyword overlap clamped into `(0, 1)`

### Code review

- `+0.057` for each newly identified true issue
- `+0.050` for a correct fix submitted after the issue has been identified
- `-0.040` per false positive finding
- `-0.100` for approving the diff while any security issue remains undetected
- Final grader score: weighted detection, fix quality, and summary correctness clamped into `(0, 1)`

## 6. Setup and local run instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI environment server locally on the Hugging Face Spaces default port:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Run inference against the local server:

```bash
set API_KEY=your_proxy_key_here
set API_BASE_URL=your_proxy_base_url_here
python inference.py --task data-triage-easy --server-url http://127.0.0.1:7860
```

For local Hugging Face Router testing, `HF_TOKEN` is also supported as a fallback when `API_KEY` is not set.

Build the container image:

```bash
docker build .
```

## 7. Baseline performance scores

Fill this section after running `inference.py` locally:

- `data-triage-easy`: 1.000
- `email-triage-medium`: 1.000
- `code-review-hard`: 0.371

## 8. HF Space URL

https://huggingface.co/spaces/itzrick/developer-workflow-env
