from envs.code_review import CodeReviewEnv
from envs.data_triage import DataTriageEnv
from envs.email_triage import EmailTriageEnv

ENV_REGISTRY = {
    "data-triage-easy": DataTriageEnv,
    "email-triage-medium": EmailTriageEnv,
    "code-review-hard": CodeReviewEnv,
}

__all__ = [
    "DataTriageEnv",
    "EmailTriageEnv",
    "CodeReviewEnv",
    "ENV_REGISTRY",
]
