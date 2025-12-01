from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .modules.utils import layer_init

class NatureCNN(nn.Module):
    def __init__(
        self, obs_dim, action_dim
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        extractors = dict()

        self.out_features = 0
        feature_size = 256

        for k in ["fetch_head_depth", "fetch_hand_depth"]:
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.obs_dim,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flatten = 9216
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors[k] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        # for state data we simply pass it through a single linear layer
        extractors["state"] = nn.Linear(42, 256)
        self.out_features += 256  # 256+256+256=768

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key == "state":
                encoded_tensor_list.append(extractor(state))
            else:
                with torch.no_grad():
                    pobs = pixels[key].float()
                    # pobs = 1 / (1 + pixels[key].float() / 400)
                    pobs = 1 - torch.tanh(pixels[key].float() / 1000)
                    if len(pobs.shape) == 5:
                        b, fs, d, h, w = pobs.shape
                        pobs = pobs.reshape(b, fs * d, h, w)
                encoded_tensor_list.append(extractor(pobs))
        return torch.cat(encoded_tensor_list, dim=1)


class CNNPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, num_action_chunks, add_value_head
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        
        self.feature_net = NatureCNN(obs_dim, action_dim)
        latent_size = self.feature_net.out_features
        if add_value_head:
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(latent_size, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, 1)),
            )
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, self.action_dim),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.action_dim) * -0.5
        )

    # def obs_encoder(self, x):
    #     return self.feature_net(x)

    # def get_value(self, x):
    #     x = self.feature_net(x)
    #     return self.critic(x)

    # def get_action(self, x, deterministic=False):
    #     x = self.feature_net(x)
    #     action_mean = self.action_head(x)
    #     if deterministic:
    #         return action_mean
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     probs = Normal(action_mean, action_std)
    #     return probs.sample()

    def predict_action_batch(
        self, env_obs, calulate_logprobs=True, calulate_values=True, **kwargs
    ):
        obs = {
            "pixels": {
                "fetch_head_depth": env_obs["images"].to("cuda"),
                "fetch_hand_depth": env_obs["wrist_images"].squeeze(1).to("cuda"),
            },
            "state": env_obs["states"].to("cuda"),
        }
        feat = self.feature_net(obs)
        
        action_mean = self.action_head(feat)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        chunk_logprobs = probs.log_prob(action)
        
        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(feat)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {
            "pixels_head_depth": obs["pixels"]["fetch_head_depth"],
            "pixels_hand_depth": obs["pixels"]["fetch_hand_depth"],
            "state": obs["state"],
        }
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        action = data["action"]
        obs = {
            "pixels": {
                "fetch_head_depth": data["pixels_head_depth"].to("cuda"),
                "fetch_hand_depth": data["pixels_hand_depth"].squeeze(1).to("cuda"),
            },
            "state": data["states"].to("cuda"),
        }

        feat = self.feature_net(obs)
        action_mean = self.action_head(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if compute_logprobs:
            logprobs = probs.log_prob(action)
        if compute_entropy:
            entropy = probs.entropy()
        if compute_values:
            values = self.value_head(obs)
        return {"logprobs": logprobs, "values": values, "entropy": entropy}
        