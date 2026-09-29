"""The four differentiable neuron structures must run, learn and stay bionic."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = __import__("torch")

from bionic_brain.nn import Axon, Dendrite, NeuronCircuit, Soma, Synapse, best_device, from_numpy, to_numpy


class StructureTest(unittest.TestCase):
    def test_numpy_and_device_interop(self):
        self.assertIn(best_device(), {"cpu", "cuda", "mps"})
        tensor = from_numpy([[1.0, 2.0]], device="cpu", requires_grad=True)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(tensor.requires_grad)
        numpy_value = to_numpy(tensor * 2)
        self.assertEqual(numpy_value.tolist(), [[2.0, 4.0]])

    def test_dendrite_is_branch_local_and_differentiable(self):
        dendrite = Dendrite(8, 4, branches=2)
        inputs = torch.randn(2, 8, 5, requires_grad=True)
        out = dendrite(inputs)
        self.assertEqual(out.shape, (2, 4, 5))
        out.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(dendrite.weight.grad.abs().sum()), 0.0)
        # Branch-local means each branch only sees its own slice of the input.
        left = torch.randn(1, 8, 3)
        right = left.clone()
        right[:, 4:, :] = 0.0
        with torch.no_grad():
            same = dendrite(torch.cat([left[:, :4], torch.zeros_like(left[:, 4:])], dim=1))
            masked = dendrite(right)
        self.assertTrue(torch.allclose(same[:, :4], masked[:, :4], atol=1e-5))

    def test_soma_spikes_above_threshold_and_has_gradient(self):
        soma = Soma(3, threshold=1.0)
        strong = torch.full((1, 3, 4), 2.0)
        spikes = soma(strong)
        self.assertGreater(float(spikes.sum()), 0.0)
        membrane_input = torch.full((1, 3, 4), 1.2, requires_grad=True)
        soma.reset()
        soma(membrane_input).sum().backward()
        self.assertIsNotNone(membrane_input.grad)
        self.assertGreater(float(membrane_input.grad.abs().sum()), 0.0)

    def test_axon_delays_and_scales(self):
        axon = Axon(2, max_delay=3)
        spikes = torch.zeros(1, 2, 6)
        spikes[0, :, 0] = 1.0
        out = axon(spikes)
        self.assertEqual(out.shape, spikes.shape)
        # A soft delay puts most energy *after* the input, not on it.
        later = float(out[0, 0, 1:].abs().sum())
        self.assertGreater(later, float(out[0, 0, 0].abs()))
        self.assertGreater(later, 0.0)

    def test_synapse_local_rule_changes_weights_without_optimiser(self):
        synapse = Synapse(4, 3, mode="local", lr=0.1)
        before = synapse.weight.detach().clone()
        pre = torch.rand(2, 4, 5)
        post = torch.rand(2, 3, 5)
        change = synapse.local_update(pre, post, dopamine=0.5)
        self.assertGreater(change, 0.0)
        self.assertFalse(torch.allclose(before, synapse.weight.detach()))

    def test_circuit_runs_and_backpropagates_through_the_structures(self):
        circuit = NeuronCircuit(8, hidden=6, branches=2, depth=2)
        inputs = torch.randn(3, 8, 7, requires_grad=True)
        out = circuit(inputs)
        self.assertEqual(out.shape, (3, 6, 7))
        out.sum().backward()
        total_grad = sum(float(p.grad.abs().sum()) for p in circuit.parameters() if p.grad is not None)
        self.assertGreater(total_grad, 0.0)

    def test_circuit_can_learn_with_local_rule_only(self):
        circuit = NeuronCircuit(4, hidden=4, branches=1, depth=1, synapse_mode="local")
        weights_before = circuit.synapses[0].weight.detach().clone()
        inputs = torch.rand(2, 4, 6)
        for _ in range(3):
            circuit(inputs, dopamine=0.2, learn=True)
        self.assertFalse(torch.allclose(weights_before, circuit.synapses[0].weight.detach()))


if __name__ == "__main__":
    unittest.main()
