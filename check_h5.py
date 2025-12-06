import h5py
import numpy as np
import imageio

h5_path = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-SetTable/open/fridge.h5"
# h5_path = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-PrepareGroceries/pick/002_master_chef_can.h5"

with h5py.File(h5_path, "r") as f:
    print("Keys/Groups in the H5 file:")
    # def print_group(name, obj):
    #     print(name)
    # f.visititems(print_group)
    # 你也可以用 list(f.keys()) 看最顶层的 keys
    # print("\nTop-level keys:", list(f.keys()))
    # breakpoint()
    # 如果你想看某个 key 的内容，比如第一个 key
    traj = f["traj_10"]
    print(traj)
    print(len(f.keys()))
    print("This is a group. Subkeys:", list(traj.keys()))
    print(traj["actions"].shape)
    print(list(traj["obs"].keys()))
    print(type(traj["obs"]["sensor_data"]["fetch_hand"]["rgb"]))  # <class 'h5py._hl.dataset.Dataset'>
    print(traj["obs"]["sensor_data"]["fetch_hand"]["rgb"].shape)  # (201, 128, 128, 3)
    
    # 读取图片数据并保存为 mp4
    rgb_data = traj["obs"]["sensor_data"]["fetch_hand"]["rgb"]
    print(f"正在读取 {rgb_data.shape[0]} 帧图片...")
    
    # 确保数据类型为 uint8，值范围在 0-255
    frames = np.array(rgb_data)
    if frames.dtype != np.uint8:
        # 如果数据是浮点数（0-1范围），转换为 uint8
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # 保存为 mp4 视频
    output_path = "output_video.mp4"
    print(f"正在保存视频到 {output_path}...")
    imageio.mimwrite(output_path, frames, fps=30, codec='libx264', quality=8)
    print(f"视频已保存到 {output_path}")
    print(list(traj["obs"]["extra"].keys()))
    print(list(traj["obs"]["sensor_param"].keys()))
    if len(f.keys()) > 0:
        first_key = list(f.keys())[0]
        data = f[first_key]
        print(f"\nFirst key: {first_key}")
        if isinstance(data, h5py.Group):
            print("This is a group. Subkeys:", list(data.keys()))
        elif isinstance(data, h5py.Dataset):
            print("This is a dataset. Shape:", data.shape)
            print("Data preview:", data[:10] if data.shape[0] > 10 else data[:])

print()