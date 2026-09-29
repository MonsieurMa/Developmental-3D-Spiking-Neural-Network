"""Differentiable neuron structures (the bridge between bionics and PyTorch).

Every module here models one of the four biological structures and states what
the backward pass approximates:

* :class:`Dendrite` - local integration over input branches (convolution is the
  population version of a dendritic receptive field)
* :class:`Soma`     - threshold/spike dynamics with a surrogate gradient (the
  backward pass approximates top-down prediction error reaching the soma)
* :class:`Axon`     - conduction delay and myelin-scaled attenuation
* :class:`Synapse`  - local plasticity (STDP / eligibility) gated by
  neuromodulators, with an optional differentiable weight

:class:`NeuronCircuit` composes the four into one unit of computation, so a
network built from it is *both* biologically structured and trainable with
backpropagation where that is the best available approximation.
"""
from .device import best_device, from_numpy, to_numpy
from .modules import Axon, Dendrite, NeuronCircuit, Soma, Synapse

__all__ = [
    "Axon", "Dendrite", "NeuronCircuit", "Soma", "Synapse",
    "best_device", "from_numpy", "to_numpy",
]
