import h5py
import hydra
import numpy as np
from tqdm import tqdm
import os
import json
from replay_training_data import load_h5_episodes
from rlinf.envs.maniskillhab.maniskillhab_env import ManiskillHABEnv

fridge_json = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-SetTable/open/fridge.json"
fridge_h5 = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-SetTable/open/fridge.h5"
episode_id = 1
@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    print("cfg.env.train.num_envs=", cfg.env.train.num_envs)
    # 从json中加载init文件等信息
    with open(fridge_json, 'r') as f:
        fridge_json_data = json.load(f)
    subtask_uid = fridge_json_data["episodes"][episode_id]["subtask_uid"]
    # NOTE(mjwei): 通过设置环境变量 SUBTASK_UID 来控制加载的init文件，以尽量不改动代码来传参。。。。。。。。
    os.environ["SUBTASK_UID"] = subtask_uid  # 替换为你需要的具体值
    # 这个很关键，设置seed用于重放
    reset_seed = fridge_json_data["episodes"][episode_id]["episode_seed"]
    
    # 从h5中初始化信息
    with h5py.File(fridge_h5, "r") as f:
        traj_key_index = list(f.keys())[episode_id]
        h5_episode = f[traj_key_index]
        actions = h5_episode["actions"]
        num_steps = actions.shape[0]  # 从demo中获取的actions是整个交互的步数
        
        # 初始化环境
        env = ManiskillHABEnv(cfg.env.train, seed_offset=0, total_num_processes=1)
        # 开始
        env.step(reset_kwargs={"seed":reset_seed})
        # 保存第一帧的图片
        env.flush_video("test-mshab-start")
        # 随机的动作
        action = np.random.random((cfg.env.train.num_envs, cfg.actor.model.action_dim))  # fetch robot's action dim = 13
        # 从环境
        success_once = False
        for step in tqdm(range(num_steps)):
            action = np.array([actions[step]])  # [num_env=1, step, action_dim]
            # breakpoint()
            extracted_obs, step_reward, terminations, truncations, infos = env.step(action)
            success_once = success_once | infos["episode"]["success_once"][0]
            # breakpoint()
            # torch.save(extracted_obs, "/mnt/mnt/public/mjwei/repo/RLinf-1111/RLinf/extracted_obs.pt")
            # if step % 100 == 0:
            #     # 保存前十步的Video
            #     env.flush_video(f"test-mshab-{step}")
                # reset
                # env.seed = list(range(0, cfg.env.train.num_envs))
                # env.seed = 0
                # env.is_start = True
                # env.step()
                # env.flush_video("test-mshab-wait")  # 保存wait 10步
                # print(step)
        breakpoint()
        print(success_once)
        env.flush_video(f"test-mshab-{step}")
    

if __name__ == "__main__":
    main()
