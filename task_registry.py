from envs import ENV_REGISTRY
from graders import GRADER_REGISTRY

TASK_REGISTRY = {
    task_name: {
        "env": ENV_REGISTRY[task_name],
        "grader": GRADER_REGISTRY[task_name],
    }
    for task_name in ENV_REGISTRY
    if task_name in GRADER_REGISTRY
}

AVAILABLE_TASKS = list(TASK_REGISTRY)
