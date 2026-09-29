"""Policy ratchet: the codebase must not grow new hand-written language rules.

``docs/TODO_ZERO_HARDCODED.md`` lists every hardcoded linguistic rule that
currently exists (H1-H12) and the plan to replace each one with a learned
mechanism. This test freezes the current violations: the allowlist may only
shrink. Adding a new word list, marker table or language-specific regex fails
the suite.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from bionic_brain import BionicBrain
from bionic_brain.config.defaults import BionicConfig
from bionic_brain.memory.learned_schemas import LearnedSchemaMemory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "bionic_brain"

# Current violations, keyed by file: constant names. Delete entries as the
# mechanisms are replaced - never add.
ALLOWED = {}

LEXICON_NAME = re.compile(r"(LEXICON|MARKERS|TOKENS|PARTICLES|WORDS|STOP|GRAMMAR|PHRASES)$", re.IGNORECASE)
MIN_ENTRIES = 3


def _string_entries(node: ast.AST) -> int:
    """How many string literals a container literal holds."""

    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return sum(1 for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str))
    if isinstance(node, ast.Dict):
        return sum(
            1 for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
        return sum(
            len(_string_elements(arg)) for arg in node.args
        )
    return 0


def _string_elements(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return [item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    return []


class NoNewLanguageLexiconTest(unittest.TestCase):
    def test_default_brain_does_not_use_symbolic_language_schemas(self):
        brain = BionicBrain(BionicConfig(seed_profile="minimal"), develop=False)
        self.assertIsInstance(brain.schemas, LearnedSchemaMemory)
        self.assertTrue(brain.schemas.use_learned_relations)
        self.assertEqual(brain.schemas.learned_markers, set())

    def test_no_new_hand_written_language_tables(self):
        violations: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            # Quarantine is explicit: default inference cannot import from here.
            if "legacy" in path.parts:
                continue
            relative = path.relative_to(SRC).as_posix()
            allowed = ALLOWED.get(relative, set())
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets: list[str] = []
                value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    value = node.value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    targets = [node.target.id]
                    value = node.value
                if value is None or not targets:
                    continue
                entries = _string_entries(value)
                if entries < MIN_ENTRIES:
                    continue
                for name in targets:
                    if LEXICON_NAME.search(name) and name not in allowed:
                        violations.append(f"{relative}:{name} ({entries} entries)")
        self.assertFalse(
            violations,
            "new hand-written language tables found; language knowledge must be learned "
            f"(see docs/TODO_ZERO_HARDCODED.md): {violations}",
        )

    def test_allowlist_only_shrinks(self):
        """If a listed violation no longer exists, the entry must be removed."""

        stale: list[str] = []
        for relative, names in ALLOWED.items():
            path = SRC / relative
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            for name in names:
                if not re.search(rf"^\s*{re.escape(name)}\s*[:=]", text, re.MULTILINE):
                    stale.append(f"{relative}:{name}")
        self.assertFalse(stale, f"allowlist entries no longer present, remove them: {stale}")


if __name__ == "__main__":
    unittest.main()
