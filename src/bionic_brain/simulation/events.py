import heapq
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass(order=True)
class SimEvent:
    time: float
    sequence: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


class EventQueue:
    """Deterministic chronological event queue for delayed spikes and tract signals."""

    def __init__(self) -> None:
        self._heap: list[SimEvent] = []
        self._sequence = 0

    def schedule(self, time: float, kind: str, **payload) -> None:
        heapq.heappush(self._heap, SimEvent(float(time), self._sequence, kind, payload))
        self._sequence += 1

    def due(self, until: float) -> list[SimEvent]:
        result: list[SimEvent] = []
        while self._heap and self._heap[0].time <= until + 1e-9:
            result.append(heapq.heappop(self._heap))
        return result

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)
