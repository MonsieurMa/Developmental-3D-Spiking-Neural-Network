import numpy as np

from ..atlas.regions import (
    MORPHOGEN_CHANNELS,
    REGIONAL_MORPHOGENS,
    REGION_BY_CODE,
)


class MorphogenField:
    """Diffusing regional identities and global AP/DV/LR axes."""

    def __init__(self, bounds, shape, diffusion=0.045, decay=0.010, inhibition=0.035):
        self.bounds = tuple(float(v) for v in bounds)
        self.shape = tuple(int(v) for v in shape)
        self.diffusion = float(diffusion)
        self.decay = float(decay)
        self.inhibition = float(inhibition)
        self.concentration = np.zeros((len(MORPHOGEN_CHANNELS), *self.shape), dtype=float)
        self.time = 0.0
        self._seed_axes()

    def _world_to_index(self, pos):
        lo = self.bounds[0::2]
        hi = self.bounds[1::2]
        fractions = [(float(pos[i]) - lo[i]) / (hi[i] - lo[i]) for i in range(3)]
        idx = [int(np.clip(round(f * (n - 1)), 0, n - 1)) for f, n in zip(fractions, self.shape)]
        return tuple(idx)

    def _seed_axes(self) -> None:
        x, y, z = np.meshgrid(
            np.linspace(self.bounds[0], self.bounds[1], self.shape[0]),
            np.linspace(self.bounds[2], self.bounds[3], self.shape[1]),
            np.linspace(self.bounds[4], self.bounds[5], self.shape[2]),
            indexing="ij",
        )
        channel = {name: i for i, name in enumerate(MORPHOGEN_CHANNELS)}
        self.concentration[channel["M_AP"]] = (y - self.bounds[2]) / 170.0
        self.concentration[channel["M_DV"]] = (z - self.bounds[4]) / 130.0
        self.concentration[channel["M_LR"]] = 1.0 - np.abs(x) / 70.0

    def secrete_region(self, region: str, amount: float = 1.0, radius_cells: int = 1) -> None:
        if region not in REGION_BY_CODE or region not in REGION_BY_CODE:
            return
        spec = REGION_BY_CODE[region]
        channel = MORPHOGEN_CHANNELS.index(spec.morphogen)
        ix, iy, iz = self._world_to_index(spec.center)
        x0, x1 = max(0, ix-radius_cells), min(self.shape[0], ix+radius_cells+1)
        y0, y1 = max(0, iy-radius_cells), min(self.shape[1], iy+radius_cells+1)
        z0, z1 = max(0, iz-radius_cells), min(self.shape[2], iz+radius_cells+1)
        self.concentration[channel, x0:x1, y0:y1, z0:z1] += amount

    def diffuse(self, dt: float = 10.0) -> None:
        c = self.concentration
        lap = np.zeros_like(c)
        lap[:, 1:-1, :, :] += c[:, :-2, :, :] + c[:, 2:, :, :] - 2.0 * c[:, 1:-1, :, :]
        lap[:, :, 1:-1, :] += c[:, :, :-2, :] + c[:, :, 2:, :] - 2.0 * c[:, :, 1:-1, :]
        lap[:, :, :, 1:-1] += c[:, :, :, :-2] + c[:, :, :, 2:] - 2.0 * c[:, :, :, 1:-1]
        # Explicit kernel is normalized for numerical stability at PRD chem_dt.
        diffusion_rate = min(0.14, self.diffusion * dt)
        decay_rate = min(0.20, self.decay * dt)
        inhibition_rate = min(0.18, self.inhibition * dt)
        c += diffusion_rate * lap - decay_rate * c
        # Mutual inhibition between all regional identity channels.
        regional = c[:len(REGIONAL_MORPHOGENS)]
        competitor = regional.sum(axis=0, keepdims=True) - regional
        regional -= inhibition_rate * competitor
        np.clip(c, 0.0, 4.0, out=c)
        self.time += dt

    def sample(self, pos) -> np.ndarray:
        return self.concentration[(slice(None), *self._world_to_index(pos))].copy()

    def gradient(self, pos, delta: float = 2.0) -> np.ndarray:
        idx = self._world_to_index(pos)
        grad = np.zeros((len(MORPHOGEN_CHANNELS), 3), dtype=float)
        for axis, offset in enumerate((-1, 1)):
            neighbor = list(idx)
            neighbor[axis] = int(np.clip(neighbor[axis] + offset, 0, self.shape[axis] - 1))
            grad[:, axis] = (self.concentration[(slice(None), *tuple(neighbor))] - self.concentration[(slice(None), *idx)]) / (delta if offset > 0 else -delta)
        return grad

    def regional_activity(self, pos) -> tuple[str, float]:
        values = self.concentration[(slice(None, len(REGIONAL_MORPHOGENS)), *self._world_to_index(pos))]
        channel = int(np.argmax(values))
        return REGIONAL_MORPHOGENS[channel], float(values[channel])

    def classify(self, pos) -> str:
        """Map dominant regional morphogen to the nearest center sharing it."""
        channel_name, _ = self.regional_activity(pos)
        best, best_distance = None, float("inf")
        px, py, pz = map(float, pos)
        for spec in REGION_BY_CODE.values():
            if spec.morphogen != channel_name:
                continue
            cx, cy, cz = spec.center
            distance = (px-cx)**2 + (py-cy)**2 + (pz-cz)**2
            if distance < best_distance:
                best, best_distance = spec.code, distance
        return best or "MB"




