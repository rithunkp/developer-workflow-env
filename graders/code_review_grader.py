from __future__ import annotations

from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _is_valid_python_snippet(snippet: str) -> bool:
    if not snippet.strip():
        return False
    try:
        compile(snippet + "\n", "<fix>", "exec")
        return True
    except SyntaxError:
        indented = "\n".join(f"    {line}" if line else "    " for line in snippet.splitlines())
        wrapped = f"def _tmp():\n{indented}\n"
        try:
            compile(wrapped, "<fix>", "exec")
            return True
        except SyntaxError:
            return False


def _normalize_finding(finding: dict[str, Any]) -> tuple[str, int]:
    return str(finding.get("issue_type", "")).strip().lower(), int(finding.get("line", -1))


def _evaluate_fix(issue: dict[str, Any], fix: dict[str, Any]) -> float:
    fixed = str(fix.get("fixed", ""))
    original = str(fix.get("original", "")).strip()
    expected_original = str(issue.get("buggy", "")).strip()
    original_matches = original == expected_original

    if not _is_valid_python_snippet(fixed):
        return 0.0

    line = issue["line"]
    if line == 6:
        if "headers.get" in fixed and "X-Token" in fixed and "args.get" not in fixed and issue["token_default"] not in fixed:
            return 1.0 if original_matches else 0.5
        if "headers.get" in fixed or "X-Token" in fixed:
            return 0.5
        return 0.0
    if line == 8:
        if "json.loads" in fixed and "eval(" not in fixed:
            return 1.0 if original_matches else 0.5
        if "json.loads" in fixed:
            return 0.5
        return 0.0
    if line == 9:
        if "retries < max_retries" in fixed:
            return 1.0 if original_matches else 0.5
        if "max_retries" in fixed and "<" in fixed:
            return 0.5
        return 0.0
    if line == 10:
        has_membership_check = "user_role in" in fixed and "admin" in fixed and "maintainer" in fixed
        has_explicit_or = "user_role ==" in fixed and "admin" in fixed and "maintainer" in fixed and 'or "maintainer"' not in fixed
        if has_membership_check or has_explicit_or:
            return 1.0 if original_matches else 0.5
        if "admin" in fixed and "maintainer" in fixed:
            return 0.5
        return 0.0
    if line == 11:
        if "cleaned_items = []" in fixed:
            return 1.0 if original_matches else 0.5
        if "cleaned_items" in fixed:
            return 0.5
        return 0.0
    if line == 12:
        if "for item in records:" in fixed:
            return 1.0 if original_matches else 0.5
        if "for" in fixed and "records" in fixed:
            return 0.5
        return 0.0
    if line == 16:
        uses_list_command = "[" in fixed and '"tar"' in fixed and '"-xf"' in fixed
        safe_subprocess = "check_output" in fixed and "shell=True" not in fixed and ("command" in fixed or uses_list_command)
        if safe_subprocess:
            return 1.0 if original_matches else 0.5
        if "check_output" in fixed and "shell=True" not in fixed:
            return 0.5
        return 0.0
    return 0.0


def _expected_summary(unresolved_issues: list[dict[str, Any]]) -> tuple[bool, str]:
    if not unresolved_issues:
        return True, "low"
    unresolved_types = {issue["issue_type"] for issue in unresolved_issues}
    if "security" in unresolved_types:
        return False, "high"
    if "logic" in unresolved_types:
        return False, "medium"
    return False, "low"


def grade_code_review(
    findings: list[dict[str, Any]],
    fixes: list[dict[str, Any]],
    summary_action: dict[str, Any] | None,
    truth_issues: list[dict[str, Any]],
) -> dict:
    issue_lookup = {(issue["issue_type"], issue["line"]): issue for issue in truth_issues}
    matched_truth_keys: set[tuple[str, int]] = set()
    false_positives = 0

    for finding in findings:
        key = _normalize_finding(finding)
        if key in issue_lookup and key not in matched_truth_keys:
            matched_truth_keys.add(key)
        else:
            false_positives += 1

    fix_scores: list[float] = []
    resolved_issue_keys: set[tuple[str, int]] = set()
    for issue in truth_issues:
        key = (issue["issue_type"], issue["line"])
        if key not in matched_truth_keys:
            continue
        issue_fix_scores = [
            _evaluate_fix(issue, fix)
            for fix in fixes
            if int(fix.get("line", -1)) == issue["line"]
        ]
        if not issue_fix_scores:
            continue
        best_score = max(issue_fix_scores)
        fix_scores.append(best_score)
        if best_score >= 1.0:
            resolved_issue_keys.add(key)

    unresolved_issues = [issue for issue in truth_issues if (issue["issue_type"], issue["line"]) not in resolved_issue_keys]
    expected_approved, expected_risk = _expected_summary(unresolved_issues)

    summary_correctness = 0.0
    if summary_action:
        approved_ok = bool(summary_action.get("approved")) == expected_approved
        risk_ok = str(summary_action.get("risk_level", "")).lower() == expected_risk
        if approved_ok and risk_ok:
            summary_correctness = 1.0
        elif approved_ok or risk_ok:
            summary_correctness = 0.5

    detection = ((len(matched_truth_keys) / 7.0) * 0.40) - (false_positives * 0.04)
    fix_quality = ((sum(fix_scores) / len(fix_scores)) * 0.35) if fix_scores else 0.0
    summary_score = summary_correctness * 0.25
    security_issues_remaining = [issue for issue in unresolved_issues if issue["issue_type"] == "security"]

    return {
        "score": _clamp(detection + fix_quality + summary_score),
        "true_positives": len(matched_truth_keys),
        "false_positives": false_positives,
        "fix_scores": fix_scores,
        "summary_correctness": summary_correctness,
        "security_issues_remaining": len(security_issues_remaining),
        "resolved_issue_keys": list(resolved_issue_keys),
        "matched_truth_keys": list(matched_truth_keys),
    }
