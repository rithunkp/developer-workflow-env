from __future__ import annotations

from typing import Iterable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_nulls(nulls: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    normalized: set[tuple[int, int]] = set()
    for cell in nulls:
        try:
            row_idx, col_idx = list(cell)
        except (TypeError, ValueError):
            continue
        normalized.add((int(row_idx), int(col_idx)))
    return normalized


def _normalize_rows(rows: Iterable[int]) -> set[int]:
    normalized: set[int] = set()
    for row_idx in rows:
        try:
            normalized.add(int(row_idx))
        except (TypeError, ValueError):
            continue
    return normalized


def grade_data_triage(action: dict, truth: dict) -> dict:
    predicted_nulls = _normalize_nulls(action.get("nulls", []))
    predicted_duplicates = _normalize_rows(action.get("duplicates", []))
    true_nulls = _normalize_nulls(truth.get("nulls", []))
    true_duplicates = _normalize_rows(truth.get("duplicates", []))

    correct_nulls = len(predicted_nulls & true_nulls)
    correct_duplicates = len(predicted_duplicates & true_duplicates)
    false_positives = len(predicted_nulls - true_nulls) + len(predicted_duplicates - true_duplicates)

    total_nulls = len(true_nulls)
    total_duplicates = len(true_duplicates)

    null_score = (correct_nulls / total_nulls) * 0.5 if total_nulls else 0.5
    duplicate_score = (correct_duplicates / total_duplicates) * 0.5 if total_duplicates else 0.5
    penalty = false_positives * 0.10
    score = _clamp(null_score + duplicate_score - penalty)

    return {
        "score": score,
        "correct_nulls": correct_nulls,
        "correct_duplicates": correct_duplicates,
        "false_positives": false_positives,
        "total_nulls": total_nulls,
        "total_duplicates": total_duplicates,
    }
