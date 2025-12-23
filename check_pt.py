import torch

extract_obs = torch.load(
    "/mnt/mnt/public/mjwei/repo/RLinf-1111/RLinf/logs/20251204-13:35:36/test_cnn/checkpoints/global_step_1/actor/model.pt",
    weights_only=False,
)
breakpoint()
