import torch

ckpt_path = "outputs/omnire_experiments/exp1/checkpoint_final.pth"
state_dict = torch.load(ckpt_path)
models_state = state_dict["models"]

if "CamPose" in models_state:
    embeds = models_state["CamPose"]["embeds.weight"]
    print(f"CamPose embeds shape: {embeds.shape}")
    print(f"Sample embed[0]: {embeds[0]}")
    
    # The 9 values likely encode rotation + translation offsets
    # Last 3 might be translation - try adding offset there
    # This is a guess - need to check the actual CamPose implementation
    translation_offset = torch.tensor([0, 0, 0, 0, 0, 0, 1.0, 0, 0], device="cuda")  # shift X
    models_state["CamPose"]["embeds.weight"] += translation_offset

torch.save(state_dict, "outputs/omnire_experiments/exp1/modified_checkpoint.pth")
