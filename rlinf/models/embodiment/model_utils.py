# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_logprobs_from_logits(logits, target):
    logprobs = -F.cross_entropy(
        logits, target=target, reduction="none"
    )  # [B, action-dim]
    return logprobs


def compute_entropy_from_logits(logits, epsilon=1e-10):
    """
    Compute entropy by logits.

    Args:
        logits: [B, vocab-size, seq-len]
    Returns:
        entropy: [B, seq-len]
    """
    all_probs = F.softmax(logits, dim=1)  # [B, vocab-size, seq-len]
    all_log_probs = torch.log(all_probs + epsilon)
    entropy = -torch.sum(all_probs * all_log_probs, dim=1)  # [B, seq-len]
    return entropy


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return np.prod(layers(x).shape)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * noise.pow(2) - log_std
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi
