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

import copy
import hydra
import torch
from omegaconf import open_dict

from rlinf.models import get_model


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    actor_model_config = cfg.actor.model
    rollout_model_config = copy.deepcopy(actor_model_config)
    with open_dict(rollout_model_config):
        rollout_model_config.precision = cfg.rollout.model.precision
        rollout_model_config.path = cfg.rollout.model.model_path
    
    model = get_model(actor_model_config)
    _train_sampling_params = {
        "temperature": cfg.algorithm.sampling_params["temperature_train"],
        "top_k": cfg.algorithm.sampling_params["top_k"],
        "top_p": cfg.algorithm.sampling_params["top_p"],
        "max_new_tokens": cfg.algorithm.length_params["max_new_token"],
        "use_cache": True,
    }
    kwargs = _train_sampling_params
    kwargs["do_sample"] = cfg.actor.model.get("do_sample", True)

    env_obs = torch.load(
        "/mnt/mnt/public_zgc/home/mjwei/repo/RLinf/outputs/extracted_obs.pt"
    )
    with torch.no_grad():
        actions, result = model.predict_action_batch(
            env_obs=env_obs,
            **kwargs,
        )

if __name__ == "__main__":
    main()
