from __future__ import annotations


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def keyword_overlap(note: str | None, reference: list[str]) -> float:
    if not reference:
        return 1.0
    if not note:
        return 0.0
    note_tokens = set(note.lower().replace("-", " ").split())
    ref_tokens = {token.lower() for token in reference}
    return len(note_tokens & ref_tokens) / len(ref_tokens)


def grade_email_triage(submissions: dict[str, dict], answers: dict[str, dict]) -> dict:
    total_emails = len(answers)
    urgent_ids = [email_id for email_id, answer in answers.items() if answer["priority"] == "urgent"]

    correct_categories = 0
    correct_priorities = 0
    note_scores: list[float] = []

    for email_id, answer in answers.items():
        submission = submissions.get(email_id, {})
        if submission.get("category") == answer["category"]:
            correct_categories += 1
        if submission.get("priority") == answer["priority"]:
            correct_priorities += 1
        if email_id in urgent_ids:
            note_scores.append(keyword_overlap(submission.get("note"), answer.get("note_keywords", [])))

    category_accuracy = (correct_categories / total_emails) * 0.40 if total_emails else 0.0
    priority_accuracy = (correct_priorities / total_emails) * 0.35 if total_emails else 0.0
    note_quality = ((sum(note_scores) / len(note_scores)) * 0.25) if note_scores else 0.0

    return {
        "score": _clamp(category_accuracy + priority_accuracy + note_quality),
        "correct_categories": correct_categories,
        "correct_priorities": correct_priorities,
        "urgent_count": len(urgent_ids),
        "note_quality": note_quality,
    }
