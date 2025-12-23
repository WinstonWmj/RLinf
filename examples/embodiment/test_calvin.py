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

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.envs.calvin.calvin_gym_env import CalvinEnv


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    import os

    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stage_num = cfg.rollout.pipeline_stage_num
    train_num_envs_per_stage = cfg.env.train.total_num_envs // _world_size // stage_num
    print("cfg.env.train.total_num_envs=", cfg.env.train.total_num_envs)
    print("train_num_envs_per_stage=", train_num_envs_per_stage)
    env = CalvinEnv(
        cfg.env.train,
        num_envs=train_num_envs_per_stage,
        seed_offset=0,
        total_num_processes=1,
    )

    # 开始
    env.is_start = True
    extracted_obs, infos = env.reset()
    env.flush_video("test-calvin-wait")  # 保存wait 10步
    a = np.random.random(
        (train_num_envs_per_stage, cfg.actor.model.action_dim)
    )  # fetch robot's action dim = 13
    for i in tqdm(range(1, 10)):
        extracted_obs, step_reward, terminations, truncations, infos = env.step(a)
        torch.save(
            extracted_obs,
            "/mnt/public/mjwei/repo/RLinf-mshab/outputs/extracted_obs_calvin.pt",
        )
        if i % 10 == 0:
            # 保存前十步的Video
            env.flush_video("test-calvin")
            # reset
            env.is_start = True
            env.step()
            env.flush_video("test-calvin-wait")  # 保存wait 10步
    env.flush_video("test-calvin")


if __name__ == "__main__":
    main()
