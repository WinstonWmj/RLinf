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

import torch.nn as nn
import torch

class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, 
        llm_dim: int, 
        proprio_dim: int,
        activation: str = "gelu",  # 'relu' or 'gelu'
    ) -> None:
        super().__init__()
        layers = []
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers.append(nn.Linear(self.proprio_dim, self.llm_dim, bias=True))
        layers.append(act())
        layers.append(nn.Linear(self.llm_dim, self.llm_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        return self.mlp(proprio)
        