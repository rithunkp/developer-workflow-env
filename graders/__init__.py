from graders.code_review_grader import grade_code_review
from graders.data_triage_grader import grade_data_triage
from graders.email_triage_grader import grade_email_triage

GRADER_REGISTRY = {
    "data-triage-easy": grade_data_triage,
    "email-triage-medium": grade_email_triage,
    "code-review-hard": grade_code_review,
}

__all__ = [
    "grade_data_triage",
    "grade_email_triage",
    "grade_code_review",
    "GRADER_REGISTRY",
]
