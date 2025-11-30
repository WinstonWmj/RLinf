import h5py

h5_path = "/mnt/mnt/public_zgc/datasets/arth-shukla/MS-HAB-PrepareGroceries/pick/002_master_chef_can.h5"

with h5py.File(h5_path, "r") as f:
    print("Keys/Groups in the H5 file:")
    def print_group(name, obj):
        print(name)
    f.visititems(print_group)
    # 你也可以用 list(f.keys()) 看最顶层的 keys
    print("\nTop-level keys:", list(f.keys()))
    breakpoint()
    # 如果你想看某个 key 的内容，比如第一个 key
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