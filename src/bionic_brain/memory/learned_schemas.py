"""Learned role/filler schema: no symbolic language fallback.

Relation candidates, query cues, transfer roles, and fillers come only from
online distribution and the differentiable binding model.  If evidence is
insufficient this object returns "unknown"; it never consults a language table.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from ..language.features import TokenStatistics
from ..language.relations import RelationDiscoverer


@dataclass(frozen=True)
class Frame:
    roles: dict[str, tuple[str, ...]]
    tokens: tuple[str, ...]

    def filler(self, role: str) -> tuple[str, ...]:
        return self.roles.get(role, ())


@dataclass(frozen=True)
class Statement:
    subject: tuple[str, ...]
    relation: str
    value: tuple[str, ...]
    tokens: tuple[str, ...]

    @property
    def subject_key(self) -> str:
        return "".join(self.subject)


@dataclass(frozen=True)
class SchemaQuery:
    kind: str
    subject: tuple[str, ...]
    ask: str
    relation: str = ""
    target: tuple[str, ...] = ()


@dataclass
class SchemaResolution:
    role: str = ""
    filler: tuple[str, ...] = ()
    matched_frame: Frame | None = None
    reason: str = "no-evidence"

    @property
    def resolved(self) -> bool:
        return bool(self.filler)


class LearnedSchemaMemory:
    """Distributional relation and binding memory used by neural-first brains."""

    def __init__(
        self,
        history: int = 12,
        *,
        statistics: TokenStatistics | None = None,
        observe_statistics: bool = True,
        relative_floor: float = 0.35,
        relation_threshold: float = 0.35,
    ) -> None:
        self.token_statistics = statistics or TokenStatistics()
        self.observe_statistics = bool(observe_statistics)
        self.relative_floor = float(relative_floor)
        self.relation_threshold = float(relation_threshold)
        self.discoverer = RelationDiscoverer()
        self.use_learned_relations = True
        self.learned_markers: set[str] = set()
        self.marker_roles: dict[str, Counter[str]] = defaultdict(Counter)
        self.marker_pair_counts: Counter[frozenset[str]] = Counter()
        self.serial_frame_counts: Counter[frozenset[str]] = Counter()
        self.learned_transfer_markers: set[str] = set()
        self.frames: deque[Frame] = deque(maxlen=history)
        self.statements: deque[Statement] = deque(maxlen=400)
        self.by_subject: dict[str, list[Statement]] = {}
        self.neighbors: dict[str, Counter[str]] = defaultdict(Counter)
        self.relation_roles: dict[str, Counter[str]] = defaultdict(Counter)
        self.observed = 0
        self.resolutions = 0
        self.observations_since_refresh = 0
        self.refresh_interval = 8
        self.reparse_needed = False

    def set_relation_source(self, discoverer: RelationDiscoverer, *, enabled: bool = True, threshold: float = 0.35) -> None:
        self.discoverer = discoverer
        self.use_learned_relations = bool(enabled)
        self.relation_threshold = float(threshold)

    def refresh_relations(self, *, relative_floor: float | None = None) -> set[str]:
        self.learned_markers = self.discoverer.marker_set(
            relative_floor=self.relative_floor if relative_floor is None else relative_floor
        )
        self._refresh_transfer_markers()
        return set(self.learned_markers)

    def is_relation(self, token: str) -> bool:
        if not self.use_learned_relations:
            return False
        if self.learned_markers:
            return token in self.learned_markers
        return float(self.discoverer.score(token)) >= self.relation_threshold

    def is_transfer(self, token: str) -> bool:
        if self.learned_transfer_markers:
            return token in self.learned_transfer_markers
        return self._is_serial_marker(token)

    def learned_relations(self, top_n: int = 12) -> list[tuple[str, float]]:
        return self.discoverer.discover(top_n)

    def _is_serial_marker(self, token: str) -> bool:
        ceiling = max((float(self.discoverer.score(item)) for item in self.learned_markers), default=0.0)
        return bool(self.learned_markers) and float(self.discoverer.score(token)) >= 0.08 * ceiling > 0.0

    def _learn_serial_statistics(self, clean: list[str]) -> None:
        positions = [index for index, token in enumerate(clean) if self.is_relation(token)]
        positions.extend(
            index for index, token in enumerate(clean)
            if index not in positions and self._is_serial_marker(token)
        )
        positions.sort()
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                if positions[right] > positions[left] + 1:
                    self.marker_pair_counts[frozenset((clean[positions[left]], clean[positions[right]]))] += 1

    def _refresh_transfer_markers(self) -> None:
        self.learned_transfer_markers = set()
        ceiling = max((float(self.discoverer.score(item)) for item in self.learned_markers), default=0.0)
        score_cache = {
            token: float(self.discoverer.score(token))
            for token in {item for pair in self.serial_frame_counts for item in pair}
        }
        for pair, count in self.serial_frame_counts.items():
            accepted = pair & self.learned_markers
            weak = {item for item in pair if score_cache.get(item, 0.0) >= 0.25 * ceiling > 0.0}
            if count >= 2 and (accepted or weak):
                self.learned_transfer_markers.update(pair)

    def _clean(self, tokens) -> list[str]:
        return [str(token) for token in tokens if token and TokenStatistics.is_word_like(str(token))]

    def observe(self, tokens) -> Frame | None:
        for name, value in (
            ("observations_since_refresh", 0),
            ("refresh_interval", 8),
            ("reparse_needed", False),
            ("serial_frame_counts", Counter()),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)
        clean = self._clean(tokens)
        if len(clean) < 3:
            return None
        if self.observe_statistics:
            self.token_statistics.observe(clean, sentence_final=True)
        self.discoverer.observe(clean)
        old_markers = set(self.learned_markers)
        self.observations_since_refresh += 1
        if self.observations_since_refresh >= self.refresh_interval:
            new_markers = self.refresh_relations()
            self.observations_since_refresh = 0
            self.reparse_needed = new_markers != old_markers
        self._learn_serial_statistics(clean)
        self._refresh_transfer_markers()

        relation_indices = [index for index, token in enumerate(clean) if self.is_relation(token)]
        if not relation_indices:
            return None
        frame = self._parse_relation(clean, relation_indices)
        self.note_neighbors(clean)
        return frame

    def _parse_relation(self, clean: list[str], relation_indices: list[int]) -> Frame | None:
        roles: dict[str, tuple[str, ...]] = {}
        if len(clean[:relation_indices[0]]) > 1:
            roles["agent"] = tuple(clean[:relation_indices[0]])
        if len(relation_indices) == 1:
            relation = clean[relation_indices[0]]
            role = self._role_for_relation(relation)
            roles[role] = tuple(clean[relation_indices[0] + 1:])
            self.marker_roles[relation][role] += 1
        else:
            first = relation_indices[0]
            second = relation_indices[1]
            roles["object"] = tuple(clean[first + 1:second])
            roles["recipient"] = tuple(clean[second + 1:])
            if roles["object"] and roles["recipient"]:
                self.serial_frame_counts[frozenset((clean[first], clean[second]))] += 1
            for index in (first, second):
                role = "object" if index == first else "recipient"
                self.marker_roles[clean[index]][role] += 1
        frame = Frame({role: filler for role, filler in roles.items() if filler}, tuple(clean))
        self.frames.append(frame)
        self.observed += 1
        self._store_statement(clean, relation_indices[0])
        return frame

    def _role_for_relation(self, relation: str) -> str:
        counts = self.relation_roles.get(relation)
        if counts:
            return counts.most_common(1)[0][0]
        # Stable opaque slot name learned from the relation's own identity.
        return f"relation:{relation}"

    def _store_statement(self, clean: list[str], marker_index: int) -> None:
        if marker_index == 0 or marker_index + 1 >= len(clean):
            return
        statement = Statement(
            subject=tuple(clean[:marker_index]),
            relation=clean[marker_index],
            value=tuple(clean[marker_index + 1:]),
            tokens=tuple(clean),
        )
        self.statements.append(statement)
        self.by_subject.setdefault(statement.subject_key, []).append(statement)

    def reparse(self, sentences) -> int:
        """Re-read stored clauses once after a marker-set revision."""

        for name, value in (
            ("observations_since_refresh", 0),
            ("refresh_interval", 8),
            ("reparse_needed", False),
            ("serial_frame_counts", Counter()),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)
        self.refresh_relations()
        self.reparse_needed = False
        self.statements.clear()
        self.by_subject.clear()
        refreshed = 0
        for sentence in sentences:
            clean = self._clean(sentence)
            relation_indices = [index for index, token in enumerate(clean) if self.is_relation(token)]
            if relation_indices:
                self._learn_serial_statistics(clean)
                self._parse_relation(clean, relation_indices)
                refreshed += 1
        self._refresh_transfer_markers()
        return refreshed

    def note_neighbors(self, tokens) -> None:
        unique = list(dict.fromkeys(self._clean(tokens)))
        for index, token in enumerate(unique):
            for other in unique[index + 1:]:
                self.neighbors[token][other] += 1
                self.neighbors[other][token] += 1

    def query(self, question_tokens) -> SchemaQuery | None:
        clean = self._clean(question_tokens)
        if len(clean) < 2:
            return None
        relation_indices = [index for index, token in enumerate(clean) if self.is_relation(token)]
        if not relation_indices:
            return None
        index = relation_indices[0]
        relation = clean[index]
        ask = self._role_for_relation(relation)
        transfer = len(relation_indices) > 1 or self.is_transfer(relation)
        return SchemaQuery(
            kind="transfer" if transfer else "statement",
            subject=tuple(clean[:index]),
            ask=ask,
            relation=relation,
            target=tuple(clean[index + 1:]) if transfer else (),
        )

    def lookup(self, query: SchemaQuery | None) -> tuple[str, ...]:
        if query is None:
            return ()
        bucket = self.by_subject.get("".join(query.subject), [])
        if not bucket:
            for statement in reversed(self.statements):
                if all(token in statement.subject for token in query.subject):
                    bucket = [statement]
                    break
        return bucket[-1].value if bucket else ()

    def expected_fillers(self, question_tokens) -> set[str]:
        query = self.query(question_tokens)
        return set(query.target) if query is not None and query.target else set()

    def resolve(self, question_tokens, context_tokens=()) -> SchemaResolution:
        query = self.query(question_tokens)
        if query is None:
            return SchemaResolution()
        filler = self.lookup(query)
        if filler:
            self.resolutions += 1
            return SchemaResolution(query.ask, filler, reason="learned-statement")
        for frame in reversed(self.frames):
            if query.subject and all(token in frame.tokens for token in query.subject):
                filler = frame.filler("recipient" if query.kind == "transfer" else query.ask)
                if filler:
                    self.resolutions += 1
                    return SchemaResolution(query.ask, filler, frame, "learned-binding")
        return SchemaResolution(query.ask)

    def guess(self, tokens, top_k: int = 3) -> tuple[str, ...]:
        clean = self._clean(tokens)
        scores: Counter[str] = Counter()
        for token in clean:
            for neighbour, count in self.neighbors.get(token, {}).items():
                if neighbour not in clean:
                    scores[neighbour] += count
        return tuple(token for token, _ in scores.most_common(top_k))

    def state(self) -> dict:
        return {
            "observed": self.observed,
            "frames": len(self.frames),
            "statements": len(self.statements),
            "subjects": len(self.by_subject),
            "markers": sorted(self.learned_markers),
            "transfer_markers": sorted(self.learned_transfer_markers),
            "resolutions": self.resolutions,
        }

    def clear_context(self) -> None:
        self.frames.clear()
