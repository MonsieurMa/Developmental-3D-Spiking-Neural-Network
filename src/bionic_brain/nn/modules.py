"""Differentiable implementations of the four neuron structures.

Design rule (see ``docs/TODO_ZERO_HARDCODED.md``): every tensor operation must
correspond to a biological structure, and every backward pass must be
interpretable as a brain-side process. Concretely:

============  ==============================  =================================
Structure     Forward (biology)               Backward (approximation of)
============  ==============================  =================================
Dendrite      branch-local integration,       error reaching a dendrite's
              branch nonlinearity (NMDA)      own inputs, not a global gradient
Soma          leaky integration + threshold   top-down prediction error at the
              spike (surrogate gradient)      soma
Axon          conduction delay, myelin gain   credit travelling back along the
                                              same pathway with the same delay
Synapse       local Hebbian/STDP update       error times presynaptic activity
              gated by neuromodulators        (the local form of the chain rule)
============  ==============================  =================================

The modules are deliberately small and readable: this is the substrate other
mechanisms (binding, sequence models, attention-as-thalamic-gain) should be built
from, instead of writing language rules by hand.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class Dendrite(nn.Module):
    """Branch-local integration with a per-branch nonlinearity.

    ``branches`` groups the input like dendritic branches, each branch has its
    own weights and its own nonlinearity (NMDA-like), and the soma receives the
    sum. A 1-D convolution over a sequence is exactly the population case: one
    kernel per branch, one receptive field per dendrite.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        branches: int = 4,
        kernel_size: int = 1,
        branch_nonlinearity: str = "softplus",
    ) -> None:
        super().__init__()
        if in_features % branches:
            raise ValueError("in_features must be divisible by branches")
        self.branches = int(branches)
        self.out_features = int(out_features)
        self.in_per_branch = in_features // branches
        self.kernel_size = int(kernel_size)
        # Grouped convolution: one group per dendritic branch, so a branch only
        # ever integrates its own inputs.
        self.weight = nn.Parameter(
            torch.randn(branches * out_features, self.in_per_branch, self.kernel_size) * 0.05
        )
        self.branch_bias = nn.Parameter(torch.zeros(branches, out_features))
        self.branch_gain = nn.Parameter(torch.ones(branches, out_features))
        self.branch_nonlinearity = branch_nonlinearity

    def _nonlinear(self, value: Tensor) -> Tensor:
        if self.branch_nonlinearity == "softplus":
            return torch.nn.functional.softplus(value)
        if self.branch_nonlinearity == "relu":
            return torch.relu(value)
        if self.branch_nonlinearity == "tanh":
            return torch.tanh(value)
        raise ValueError(f"unknown branch nonlinearity: {self.branch_nonlinearity}")

    def forward(self, inputs: Tensor) -> Tensor:
        """``inputs``: (batch, in_features, length) -> (batch, out_features, length)."""

        batch, features, length = inputs.shape
        if features != self.branches * self.in_per_branch:
            raise ValueError("input feature count does not match the dendrite layout")
        padded = torch.nn.functional.pad(inputs, (self.kernel_size - 1, 0))
        branch_out = torch.nn.functional.conv1d(padded, self.weight, groups=self.branches)
        branch_out = branch_out / self.in_per_branch ** 0.5
        branch_out = branch_out.view(batch, self.branches, self.out_features, branch_out.shape[-1])
        branch_out = branch_out + self.branch_bias.unsqueeze(0).unsqueeze(-1)
        branch_out = self.branch_gain.unsqueeze(0).unsqueeze(-1) * self._nonlinear(branch_out)
        return branch_out.sum(dim=1)


class Soma(nn.Module):
    """Leaky integration with a spiking threshold and a surrogate gradient.

    The forward pass is a leaky integrator that emits a spike when the membrane
    crosses the threshold. The backward pass uses a surrogate derivative, i.e.
    the error is transmitted as if the threshold were soft - the network keeps a
    spiking forward path while remaining trainable.
    """

    def __init__(self, features: int, *, tau: float = 0.9, threshold: float = 1.0, surrogate_width: float = 1.0) -> None:
        super().__init__()
        self.features = int(features)
        self.tau = float(tau)
        self.threshold = nn.Parameter(torch.full((features,), float(threshold)))
        self.surrogate_width = float(surrogate_width)
        self.register_buffer("membrane", torch.zeros(1, features, 1))

    def reset(self, batch: int = 1, length: int = 1) -> None:
        # The membrane is a scalar state per neuron, not a sequence.
        self.membrane = torch.zeros(batch, self.features, 1, device=self.threshold.device)

    def forward(self, current: Tensor, *, reset: bool = True) -> Tensor:
        """``current``: (batch, features, length) -> spikes (batch, features, length)."""

        batch, features, length = current.shape
        if reset or self.membrane.shape[0] != batch:
            self.reset(batch, length)
        spikes = torch.zeros_like(current)
        membrane = self.membrane
        for step in range(length):
            membrane = self.tau * membrane + current[..., step:step + 1]
            spike = self._surrogate_spike(membrane)
            membrane = membrane - spike * self.threshold.view(1, -1, 1)
            spikes[..., step:step + 1] = spike
        self.membrane = membrane
        return spikes

    def _surrogate_spike(self, membrane: Tensor) -> Tensor:
        threshold = self.threshold.view(1, -1, 1)
        hard = (membrane >= threshold).float()
        # Straight-through with a triangular window: forward is a spike,
        # backward behaves like a soft threshold.
        soft = torch.clamp((membrane - threshold) / self.surrogate_width + 0.5, 0.0, 1.0)
        return hard.detach() - soft.detach() + soft


class Axon(nn.Module):
    """Conduction delay with myelin-scaled gain.

    Delays are implemented as a learnable soft lag (a small depthwise kernel)
    per feature; myelin scales how much charge arrives. The backward pass
    travels back through the same lag, so credit assignment respects conduction
    time instead of ignoring it.
    """

    def __init__(self, features: int, *, max_delay: int = 3) -> None:
        super().__init__()
        self.features = int(features)
        self.max_delay = int(max_delay)
        # Initialise to a pass-through with one step of delay.
        kernel = torch.zeros(features, 1, self.max_delay)
        kernel[:, 0, min(1, self.max_delay - 1)] = 1.0
        self.kernel = nn.Parameter(kernel)
        self.myelin = nn.Parameter(torch.ones(features))

    def forward(self, spikes: Tensor) -> Tensor:
        if spikes.shape[1] != self.features:
            raise ValueError("axon feature count mismatch")
        weight = torch.softmax(self.kernel, dim=-1)
        padded = torch.nn.functional.pad(spikes, (self.max_delay - 1, 0))
        delayed = torch.nn.functional.conv1d(padded, weight, groups=self.features)
        return delayed * torch.sigmoid(self.myelin).view(1, -1, 1)


class Synapse(nn.Module):
    """Local plasticity with a neuromodulator gate.

    Two update modes are supported:

    * ``local``  - the Hebbian/STDP rule used elsewhere in the project:
      ``dw = lr * pre * post`` (trace-gated), optionally scaled by dopamine;
    * ``backprop`` - the ordinary gradient, i.e. the local rule's global
      approximation.

    Keeping both makes the biological claim testable: if the local rule reaches
    the same performance as backprop on a task, the mechanism is doing the work,
    not the optimiser.
    """

    def __init__(self, in_features: int, out_features: int, *, mode: str = "local", lr: float = 0.01) -> None:
        super().__init__()
        if mode not in {"local", "backprop"}:
            raise ValueError("mode must be 'local' or 'backprop'")
        self.mode = mode
        self.lr = float(lr)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.eligibility = torch.zeros_like(self.weight)

    def forward(self, pre: Tensor, *, dopamine: float = 0.0) -> Tensor:
        """``pre``: (batch, in_features, length) -> postsynaptic current."""

        return torch.einsum("bi l,oi->bo l".replace(" ", ""), pre, self.weight)

    @torch.no_grad()
    def local_update(self, pre: Tensor, post: Tensor, *, dopamine: float = 0.0) -> float:
        """Apply the local rule and return the mean absolute change."""

        pre_rate = pre.mean(dim=(0, 2)).clamp(min=0.0)
        post_rate = post.mean(dim=(0, 2)).clamp(min=0.0)
        hebbian = torch.outer(post_rate, pre_rate)
        self.eligibility = 0.9 * self.eligibility + hebbian
        delta = self.lr * self.eligibility * (1.0 + float(dopamine))
        self.weight.add_(delta)
        return float(delta.abs().mean())


class NeuronCircuit(nn.Module):
    """Dendrite -> Soma -> Axon -> Synapse, the four structures in one unit.

    ``depth`` repeats the circuit so a stack still reads as "cortical column of
    neurons" rather than an opaque multi-layer perceptron.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        *,
        branches: int = 4,
        depth: int = 1,
        synapse_mode: str = "local",
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.dendrites = nn.ModuleList()
        self.somas = nn.ModuleList()
        self.axons = nn.ModuleList()
        self.synapses = nn.ModuleList()
        current = int(in_features)
        for _ in range(self.depth):
            branch_count = branches if current % branches == 0 else 1
            self.dendrites.append(Dendrite(current, hidden, branches=branch_count))
            self.somas.append(Soma(hidden))
            self.axons.append(Axon(hidden))
            self.synapses.append(Synapse(hidden, hidden, mode=synapse_mode))
            current = hidden

    def forward(self, inputs: Tensor, *, dopamine: float = 0.0, learn: bool = False) -> Tensor:
        """``inputs``: (batch, in_features, length) -> spikes (batch, hidden, length)."""

        signal = inputs
        for dendrite, soma, axon, synapse in zip(self.dendrites, self.somas, self.axons, self.synapses):
            current = dendrite(signal)
            spikes = soma(current)
            delayed = axon(spikes)
            signal = synapse(delayed, dopamine=dopamine)
            if learn and synapse.mode == "local":
                synapse.local_update(delayed, spikes, dopamine=dopamine)
        return signal
