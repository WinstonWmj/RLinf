import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.envs.maniskillhab.maniskillhab_env import ManiskillHABEnv


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
    env = ManiskillHABEnv(
        cfg.env.train,
        num_envs=train_num_envs_per_stage,
        seed_offset=0,
        total_num_processes=1,
    )

    # 开始
    env.is_start = True
    extracted_obs, infos = env.reset()
    env.flush_video("test-mshab-wait")  # 保存wait 10步
    a = np.random.random(
        (train_num_envs_per_stage, cfg.actor.model.action_dim)
    )  # fetch robot's action dim = 13
    for i in tqdm(range(1, 30)):
        extracted_obs, step_reward, terminations, truncations, infos = env.step(a)
        torch.save(
            extracted_obs, "/mnt/mnt/public_zgc/home/mjwei/repo/RLinf/extracted_obs.pt"
        )
        if i % 10 == 0:
            # 保存前十步的Video
            env.flush_video("test-mshab")
            # reset
            env.is_start = True
            env.step()
            env.flush_video("test-mshab-wait")  # 保存wait 10步
    env.flush_video("test-mshab")


if __name__ == "__main__":
    main()
