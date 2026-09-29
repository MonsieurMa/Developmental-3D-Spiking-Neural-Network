"""PRD-focused smoke test for the modular implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from bionic_brain import BionicBrain
from bionic_brain.atlas.regions import REGION_CODES, REGION_SPECS

random_seed = 1907
brain = BionicBrain(seed=random_seed)
assert len(REGION_SPECS) >= 30
assert all(brain.regions[code].neurons for code in REGION_CODES)
assert brain.neurons[0].position[0] >= -70.0 and brain.neurons[0].position[0] <= 70.0
assert brain.morphogens.concentration.shape[0] == 13
assert len(brain.tracts.tracts) >= 9
assert all(tract.formed for tract in brain.tracts.tracts)

first = brain.morphogens.sample((0.0, 0.0, 0.0)).copy()
for _ in range(3):
    brain.morphogens.diffuse(10.0)
assert not np.allclose(first, brain.morphogens.sample((0.0, 0.0, 0.0)))

identity, strength = brain.morphogens.regional_activity((0.0, 0.0, 0.0))
assert identity == "M_BS" and strength > 0.0
gradient = brain.morphogens.gradient((30.0, 30.0, 30.0))
assert gradient.shape == (13, 3)

brain.ensure_word("你")
assert brain.sdr.sensory_indices("你") != brain.sdr.motor_indices("你")
brain.input_text("你 好")
assert brain.stats["spikes"] > 0
assert brain.active_context

answer = brain.learn_pair("你 是 谁", "我 是 人")
assert answer == ["我", "是", "人"]
output = brain.respond("你 是 谁", max_len=3)
assert output and output != ["<unk>"]

before = brain.synapses[-1].weight
brain.reward(0.8)
assert brain.modulators.concentrations["DA"] > 0.0

brain.step(2.0)
state_path = Path("_bionic_smoke.pkl")
brain.save(state_path)
spikes = brain.stats["spikes"]
words = len(brain.bindings)
loaded = BionicBrain.load(state_path)
assert words == len(loaded.bindings)
loaded.step(0.2)
assert loaded.stats["spikes"] >= spikes
replay = loaded.sleep(replay_count=2)
assert replay["replayed"] >= 0
state_path.unlink(missing_ok=True)

print("BionicBrain smoke: PASS")
print(f"regions={len(REGION_SPECS)} seeds={len(brain.neurons)} synapses={len(brain.synapses)} words={words} output={output}")
