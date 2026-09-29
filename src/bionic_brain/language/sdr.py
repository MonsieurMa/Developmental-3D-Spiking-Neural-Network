import hashlib


class SDRSpace:
    """Stable sparse distributed representations generated from token identity."""

    def __init__(self, dimension: int = 10_000, k: int = 10, seed: int = 0x424F):
        self.dimension = int(dimension)
        self.k = int(k)
        self.seed = int(seed)
        self.vocabulary: dict[str, int] = {}

    def ensure_word(self, token: str) -> int:
        token = str(token)
        if token not in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary)
        return self.vocabulary[token]

    def indices(self, token: str, k: int | None = None, salt: str = "sdr") -> tuple[int, ...]:
        count = min(self.dimension, int(k or self.k))
        out: list[int] = []
        counter = 0
        while len(out) < count:
            payload = f"{self.seed}:{salt}:{token}:{counter}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            for offset in range(0, len(digest), 4):
                value = int.from_bytes(digest[offset:offset+4], "big") % self.dimension
                if value not in out:
                    out.append(value)
                    if len(out) == count:
                        break
            counter += 1
        return tuple(out)

    def sensory_indices(self, token: str) -> tuple[int, ...]:
        return self.indices(token, salt="wernicke")

    def motor_indices(self, token: str) -> tuple[int, ...]:
        return self.indices(token, self.k and self.k, salt="broca")

    def context_indices(self, token: str) -> tuple[int, ...]:
        return self.indices(token, 4, salt="dlpfc")

    def similarity(self, left: tuple[int, ...], right: tuple[int, ...]) -> float:
        a, b = set(left), set(right)
        union = len(a | b)
        return len(a & b) / union if union else 0.0
