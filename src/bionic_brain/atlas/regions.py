from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RegionSpec:
    code: str
    name: str
    center: tuple[float, float, float]
    radius: float
    lobe: str
    morphogen: str
    model: str = "lif"
    source: bool = True


# These are the 33 named centers given by the PRD coordinate table. Bilateral
# centers remain one developmental target in this minimal anatomical atlas.
REGION_SPECS: tuple[RegionSpec, ...] = (
    RegionSpec("MB", "中脑", (0, 0, 0), 8, "brainstem", "M_BS", "hh"),
    RegionSpec("Pons", "脑桥", (0, -10, -10), 10, "brainstem", "M_BS"),
    RegionSpec("Med", "延髓", (0, -15, -22), 8, "brainstem", "M_BS"),
    RegionSpec("Thal", "丘脑", (0, 5, 18), 12, "diencephalon", "M_THAL", "hh"),
    RegionSpec("Hypo", "下丘脑", (0, 10, 2), 8, "diencephalon", "M_BS"),
    RegionSpec("Stri", "纹状体", (20, 15, 22), 10, "basal-ganglia", "M_BG"),
    RegionSpec("GP", "苍白球", (15, 5, 18), 8, "basal-ganglia", "M_BG"),
    RegionSpec("SN", "黑质", (0, -5, -3), 6, "basal-ganglia", "M_BG"),
    RegionSpec("STN", "丘脑底核", (10, -3, 2), 5, "basal-ganglia", "M_BG"),
    RegionSpec("CB", "小脑", (0, -55, -28), 25, "cerebellum", "M_CB"),
    RegionSpec("M1", "初级运动皮层", (5, 5, 68), 8, "frontal", "M_FRONT", "hh"),
    RegionSpec("PMC", "前运动皮层", (5, 18, 68), 8, "frontal", "M_FRONT"),
    RegionSpec("SMA", "辅助运动区", (5, 0, 72), 6, "frontal", "M_FRONT"),
    RegionSpec("Broca", "布洛卡区", (-40, 50, 28), 10, "frontal", "M_FRONT", "hh"),
    RegionSpec("DLPFC", "背外侧前额叶", (25, 60, 50), 14, "frontal", "M_FRONT"),
    RegionSpec("VMPFC", "腹内侧前额叶", (10, 72, 22), 10, "frontal", "M_FRONT"),
    RegionSpec("OFC", "眶额皮层", (18, 68, 12), 10, "frontal", "M_FRONT"),
    RegionSpec("S1", "初级体感皮层", (5, -25, 68), 8, "parietal", "M_PAR"),
    RegionSpec("SMG", "缘上回", (-45, -22, 48), 10, "parietal", "M_PAR"),
    RegionSpec("AG", "角回", (-45, -42, 42), 10, "parietal", "M_PAR"),
    RegionSpec("SPL", "顶上小叶", (20, -32, 72), 10, "parietal", "M_PAR"),
    RegionSpec("IPS", "顶内沟", (15, -28, 60), 8, "parietal", "M_PAR"),
    RegionSpec("A1", "初级听觉皮层", (52, 5, 12), 8, "temporal", "M_TEMP"),
    RegionSpec("Wern", "威尔尼克区", (-52, -12, 18), 12, "temporal", "M_TEMP", "hh"),
    RegionSpec("MTG", "颞中回", (55, -15, -5), 12, "temporal", "M_TEMP"),
    RegionSpec("IT", "颞下回", (58, -12, -18), 10, "temporal", "M_TEMP"),
    RegionSpec("Hipp", "海马体", (32, -5, -28), 12, "temporal", "M_HIPP", "hh"),
    RegionSpec("EC", "内嗅皮层", (38, -18, -32), 10, "temporal", "M_HIPP"),
    RegionSpec("Amy", "杏仁核", (32, 12, -22), 8, "temporal", "M_AMY", "hh"),
    RegionSpec("V1", "初级视皮层", (0, -78, 28), 10, "occipital", "M_OCC"),
    RegionSpec("V2", "V2", (0, -72, 38), 8, "occipital", "M_OCC"),
    RegionSpec("V4", "V4", (22, -68, 22), 8, "occipital", "M_OCC"),
    RegionSpec("MT", "MT/V5", (32, -62, 18), 8, "occipital", "M_OCC"),
)

REGION_BY_CODE: dict[str, RegionSpec] = {r.code: r for r in REGION_SPECS}
REGION_CODES: tuple[str, ...] = tuple(REGION_BY_CODE)

REGIONAL_MORPHOGENS = (
    "M_FRONT", "M_PAR", "M_TEMP", "M_OCC", "M_THAL", "M_BG",
    "M_CB", "M_BS", "M_HIPP", "M_AMY",
)
GLOBAL_MORPHOGENS = ("M_AP", "M_DV", "M_LR")
MORPHOGEN_CHANNELS = REGIONAL_MORPHOGENS + GLOBAL_MORPHOGENS


# Nine major white-matter buses from PRD section 4.
@dataclass(frozen=True)
class PathwaySpec:
    name: str
    source: str
    target: str
    delay: float
    attenuation: float
    conduction_velocity: float = 3.2


PATHWAYS: tuple[PathwaySpec, ...] = (
    PathwaySpec("AF", "Wern", "Broca", 20.0, 0.90),
    PathwaySpec("SLF", "DLPFC", "SPL", 30.0, 0.85),
    PathwaySpec("SLF-WT", "SPL", "Wern", 30.0, 0.85),
    PathwaySpec("ILF", "V1", "IT", 27.0, 0.90),
    PathwaySpec("UF", "MTG", "DLPFC", 35.0, 0.85),
    PathwaySpec("CC", "Broca", "Wern", 15.0, 0.95),
    PathwaySpec("Fornix", "Hipp", "Hypo", 20.0, 0.90),
    PathwaySpec("Thalamic-Radiation", "Thal", "V1", 10.0, 0.95),
    PathwaySpec("Corticospinal", "M1", "MB", 30.0, 0.90),
    PathwaySpec("Cerebellar-Peduncle", "CB", "Thal", 22.0, 0.90),
)


def euclidean(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def find_region_at(pos: tuple[float, float, float]) -> str | None:
    """Return the nearest anatomical center containing pos, if any."""
    best, best_distance = None, math.inf
    for spec in REGION_SPECS:
        distance = euclidean(pos, spec.center)
        if distance <= spec.radius and distance < best_distance:
            best, best_distance = spec.code, distance
    return best
