import h5py
import hydra
import numpy as np
from tqdm import tqdm
import os
import json
from rlinf.envs.maniskillhab.maniskillhab_env import ManiskillHABEnv

fridge_json = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-SetTable/open/fridge.json"
fridge_h5 = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-SetTable/open/fridge.h5"
episode_id = 1
@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # 从json中加载init文件等信息
    with open(fridge_json, 'r') as f:
        fridge_json_data = json.load(f)
    # 从h5中初始化信息
    with h5py.File(fridge_h5, "r") as f:
        for episode_id in range(1000):
            subtask_uid = fridge_json_data["episodes"][episode_id]["subtask_uid"]
            # NOTE(mjwei): 通过设置环境变量 SUBTASK_UID 来控制加载的init文件，以尽量不改动代码来传参。。。。。。。。
            os.environ["SUBTASK_UID"] = subtask_uid  # 替换为你需要的具体值
            # 这个很关键，设置seed用于重放
            reset_seed = fridge_json_data["episodes"][episode_id]["episode_seed"]
        
            traj_key_index = list(f.keys())[episode_id]
            h5_episode = f[traj_key_index]
            actions = h5_episode["actions"]
            num_steps = actions.shape[0]  # 从demo中获取的actions是整个交互的步数
            
            # 初始化环境
            env = ManiskillHABEnv(cfg.env.train, seed_offset=0, total_num_processes=1)
            # 开始
            env.step(reset_kwargs={"seed":reset_seed})
            # # 保存第一帧的图片
            # env.flush_video(f"test-mshab-start-episode{episode_id}")
            # # 随机的动作
            # action = np.random.random((cfg.env.train.num_envs, cfg.actor.model.action_dim))  # fetch robot's action dim = 13
            success_once = False
            for step in tqdm(range(num_steps)):
                action = np.array([actions[step]])  # [num_env=1, step, action_dim]
                extracted_obs, step_reward, terminations, truncations, infos = env.step(action)
                success_once = success_once | infos["episode"]["success_once"][0]
            success_at_end = infos["episode"]["success_at_end"]
            print(success_once)
            # 追加到txt文件
            with open("success_once_result.txt", "a") as fp:
                fp.write(f"success_once={success_once};success_at_end={success_at_end}\n")
            env.flush_video(f"test-mshab-episode{episode_id}-{step}")

if __name__ == "__main__":
    main()
