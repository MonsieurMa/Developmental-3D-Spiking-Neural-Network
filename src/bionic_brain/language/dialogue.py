"""Dialogue cue normalization from structural and distributional evidence."""
from __future__ import annotations

from .features import TokenStatistics


def core_dialogue_tokens(
    tokens: list[str],
    statistics: TokenStatistics | None = None,
    *,
    observe: bool = True,
) -> list[str]:
    """Return the stable tokens used for neural cue formation.

    Punctuation is a structural, script-independent boundary.  A sentence-final
    particle is learned from how often a token occupies that position; no
    language names a particle.
    """
    if statistics is not None and observe:
        statistics.observe(tokens, sentence_final=True)
    clean = [
        token for token in tokens
        if token and all(char.isalnum() for char in token)
    ]
    while len(clean) > 1 and statistics is not None and statistics.is_final_particle(clean[-1]):
        clean.pop()
    return clean if clean else list(tokens)
