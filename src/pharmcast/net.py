"""The PharmCast network.

A plain multilayer perceptron: SMILES features in, one logit per fingerprint
slot out. There is no sigmoid here -- the loss is applied to logits during
training and `model.predict_bits` applies the sigmoid at inference.

This definition is vendored verbatim from the trainer so that a released
checkpoint loads without the training tree present. Do not change the layer
construction: `state_dict` keys are positional (`net.0.weight`, `net.2.weight`,
...) and any inserted or reordered module silently breaks loading of every
existing checkpoint.
"""
from __future__ import annotations

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))   # logits, no sigmoid here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
