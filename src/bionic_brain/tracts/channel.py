from dataclasses import dataclass, field
import math

from ..atlas.regions import PATHWAYS, REGION_BY_CODE


@dataclass
class Tract:
    name: str
    source: str
    target: str
    delay: float
    attenuation: float
    conduction_velocity: float = 3.2
    length: float = 0.0
    activity: float = 0.0
    myelin: float = 0.2
    formed: bool = False

    def develop(self) -> None:
        a = REGION_BY_CODE[self.source].center
        b = REGION_BY_CODE[self.target].center
        self.length = math.sqrt(sum((float(x)-float(y))**2 for x, y in zip(a, b)))
        self.delay = max(5.0, self.length / self.conduction_velocity)
        self.formed = True

    def transmit(self, signal: float) -> float:
        self.activity = min(2.0, self.activity + 0.08)
        self.delay = max(5.0, self.delay * (1.0 - 0.002 * self.activity))
        self.myelin = min(1.0, self.myelin + 0.005)
        return float(signal) * self.attenuation


class TractSystem:
    def __init__(self) -> None:
        self.tracts: list[Tract] = [
            Tract(p.name, p.source, p.target, p.delay, p.attenuation, p.conduction_velocity)
            for p in PATHWAYS
        ]
        for tract in self.tracts:
            tract.develop()

    def outgoing(self, region: str) -> list[Tract]:
        return [tract for tract in self.tracts if tract.source == region and tract.formed]

    def incoming(self, region: str) -> list[Tract]:
        return [tract for tract in self.tracts if tract.target == region and tract.formed]

    def route(self, source: str, value: float, schedule):
        """Schedule signals on every tract leaving source; schedule is a callable."""
        routed = []
        for tract in self.outgoing(source):
            strength = tract.transmit(value)
            schedule(tract.target, tract.delay, strength, tract.name)
            routed.append((tract.target, tract.delay, strength, tract.name))
        return routed

    def state(self) -> list[dict]:
        return [tract.__dict__.copy() for tract in self.tracts]
