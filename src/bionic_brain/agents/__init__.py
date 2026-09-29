"""Dedicated trainer-side agents for interactive Brian curricula."""
from .llm_teacher import (
    ChatTeacherClient,
    LlmTeacherAgent,
    TeacherConfig,
    TeacherUnavailable,
    build_teacher,
)

__all__ = [
    "ChatTeacherClient",
    "LlmTeacherAgent",
    "TeacherConfig",
    "TeacherUnavailable",
    "build_teacher",
]
