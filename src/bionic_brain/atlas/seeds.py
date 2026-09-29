from dataclasses import dataclass
import math
import random

from ..config.defaults import seed_counts
from .regions import REGION_BY_CODE, REGION_CODES


@dataclass
class SeedCell:
    position: tuple[float, float, float]
    region: str
    role: str


def _sample_in_sphere(rng: random.Random, center, radius):
    while True:
        p = tuple(rng.uniform(-radius, radius) for _ in range(3))
        if math.sqrt(sum(v * v for v in p)) <= radius:
            return tuple(center[i] + p[i] for i in range(3))


def build_seed_plan(profile: str, rng: random.Random) -> list[SeedCell]:
    """Build independent regional progenitor pools.

    Regional source cells secrete identity signals, undifferentiated cells
    choose fate from local morphogen gradients, and glial progenitors provide
    metabolic/myelination support.
    """
    per_region = seed_counts(profile, len(REGION_CODES))
    source_n = max(1, round(per_region * 0.10))
    glial_n = max(1 if per_region >= 3 else 0, round(per_region * 0.20))
    undiff_n = max(0, per_region - source_n - glial_n)
    plan: list[SeedCell] = []
    for code in REGION_CODES:
        spec = REGION_BY_CODE[code]
        for _ in range(source_n):
            plan.append(SeedCell(_sample_in_sphere(rng, spec.center, spec.radius), code, "source"))
        for _ in range(undiff_n):
            plan.append(SeedCell(_sample_in_sphere(rng, spec.center, spec.radius), code, "undiff"))
        for _ in range(glial_n):
            plan.append(SeedCell(_sample_in_sphere(rng, spec.center, spec.radius), code, "glia"))
    return plan
