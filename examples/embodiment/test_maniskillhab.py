import hydra
import torch
import numpy as np
from tqdm import tqdm

from rlinf.envs.maniskillhab.maniskillhab_env import ManiskillHABEnv


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    print("cfg.env.train.num_envs=", cfg.env.train.num_envs)
    env = ManiskillHABEnv(cfg.env.train, seed_offset=0, total_num_processes=1)
    
    # 开始
    env.is_start = True
    env.step()
    env.flush_video("test-mshab-wait")  # 保存wait 10步
    a = np.random.random((cfg.env.train.num_envs, cfg.actor.model.action_dim))  # fetch robot's action dim = 13
    for i in tqdm(range(1, 30)):
        extracted_obs, step_reward, terminations, truncations, infos = env.step(a)
        torch.save(extracted_obs, "/mnt/mnt/public_zgc/home/mjwei/repo/RLinf/extracted_obs.pt")
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
