import torch
import torch.nn as nn

from rlinf.models.embodiment.model_utils import (
    gaussian_logprob,
    get_out_shape,
    squash,
    weight_init,
)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.out_dim = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SharedCNN(nn.Module):
    def __init__(self, pixel_obs_shape, features, filters, strides, padding):
        super().__init__()
        assert len(pixel_obs_shape) == 3

        in_features = [pixel_obs_shape[0]] + features
        out_features = features
        layers = []
        for i, (in_f, out_f, filter_, stride) in enumerate(
            zip(in_features, out_features, filters, strides)
        ):
            layers.append(nn.Conv2d(in_f, out_f, filter_, stride, padding=padding))
            if i < len(filters) - 1:
                layers.append(nn.ReLU())
        layers.append(Flatten())
        self.layers = nn.Sequential(*layers)

        self.out_dim = get_out_shape(pixel_obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, pixels: torch.Tensor):
        # NOTE (arth): a bit unclean? basically with we get concat'd pixel images,
        # it's easier to squash them here (once) than when actor/critic called in Algo
        # (multiple times). generally better to keep the modules separate from
        # this sort of logic, but this is easier
        with torch.no_grad():
            if len(pixels.shape) == 5:
                b, fs, d, h, w = pixels.shape
                pixels = pixels.view(b, fs * d, h, w).contiguous()
        return self.layers(pixels)


class Encoder(nn.Module):
    """Convolutional encoder of pixels and state observations."""

    def __init__(
        self,
        cnns: nn.ModuleDict,
        pixels_projections: nn.ModuleDict,
        state_projection: RLProjection,
    ):
        super().__init__()
        self.cnns = cnns
        self.pixels_projections = pixels_projections
        self.state_projection = state_projection
        self.out_dim = (
            sum([pix_proj.out_dim for pix_proj in self.pixels_projections.values()])
            + state_projection.out_dim
        )

    def forward(self, pixels, state, detach=False):
        pencs = [(k, cnn(pixels[k])) for k, cnn in self.cnns.items()]
        if detach:
            pencs = [(k, p.detach()) for k, p in pencs]
        pixels = torch.cat([self.pixels_projections[k](p) for k, p in pencs], dim=1)
        state = self.state_projection(state)
        return torch.cat([pixels, state], dim=1)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, encoder, action_dim, hidden_dims, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder

        in_dims = [self.encoder.out_dim] + hidden_dims
        out_dims = hidden_dims + [2 * action_dim]
        mlp_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(weight_init)

    def forward(
        self, pixels, state, compute_pi=True, compute_log_pi=True, detach=False
    ):
        x = self.encoder(pixels, state, detach=detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        # NOTE(mjwei): 训练需要随机性与 logprob 支持梯度/探索，用采样 pi；评估想看策略均值的稳定效果，用确定性的 mu。
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return (
            mu,
            pi,
            log_pi,
            log_std,
        )  # greedy_action, action, logprob, log_std(to get entropy)


class CNNPPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, num_action_chunks, add_value_head=False):
        super().__init__()
        actor_hidden_dims = [256, 256, 256]
        encoder_pixels_feature_dim = 50
        encoder_state_feature_dim = 50
        cnn_features = [32, 64, 128, 256]
        cnn_filters = [3, 3, 3, 3]
        cnn_strides = [2, 2, 2, 2]
        cnn_padding = "valid"
        actor_log_std_min = -20
        actor_log_std_max = 2
        self.obs_dim = 42
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        pixels_obs_space = {
            "fetch_head_depth": (3, 128, 128),
            "fetch_hand_depth": (3, 128, 128),
        }

        shared_cnns: nn.ModuleDict = nn.ModuleDict(
            (
                k,
                SharedCNN(
                    pixel_obs_shape=shape,
                    features=cnn_features,
                    filters=cnn_filters,
                    strides=cnn_strides,
                    padding=cnn_padding,
                ),
            )
            for k, shape in pixels_obs_space.items()
        )

        actor_encoder = Encoder(
            cnns=shared_cnns,
            pixels_projections=nn.ModuleDict(
                (
                    k,
                    RLProjection(
                        cnn.out_dim,
                        encoder_pixels_feature_dim,
                    ),
                )
                for k, cnn in shared_cnns.items()
            ),
            state_projection=RLProjection(
                self.obs_dim,
                encoder_state_feature_dim,
            ),
        )
        self.actor = Actor(
            encoder=actor_encoder,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            log_std_min=actor_log_std_min,
            log_std_max=actor_log_std_max,
        )

    def predict_action_batch(
        self, env_obs, calulate_logprobs=True, calulate_values=True, **kwargs
    ):
        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype
        pixel_obs = {
            "fetch_head_depth": env_obs["images"].to(device=device, dtype=precision),
            "fetch_hand_depth": env_obs["wrist_images"]
            .squeeze(1)
            .to(device=device, dtype=precision),
        }
        state_obs = env_obs["states"].to(device=device, dtype=precision)
        greedy_action, action, logprob, _ = self.actor(pixel_obs, state_obs)
        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        chunk_logprobs = (
            logprob.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        forward_inputs = {
            "images": env_obs["images"],
            "wrist_images": env_obs["wrist_images"],
            "state": env_obs["states"],
            "action": action,
        }
        result = {
            "prev_logprobs": chunk_logprobs,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype
        pixel_obs = {
            "fetch_head_depth": data["images"].to(device=device, dtype=precision),
            "fetch_hand_depth": data["wrist_images"]
            .squeeze(1)
            .to(device=device, dtype=precision),
        }
        state_obs = data["state"].to(device=device, dtype=precision)
        _, _, logprob, _ = self.actor(pixel_obs, state_obs)
        ret_dict = {}
        if compute_logprobs:
            ret_dict["logprobs"] = logprob.to(dtype=torch.float32)
        return ret_dict
