import torch

ckpt_path = "outputs/omnire_experiments/exp1/checkpoint_final.pth"
state_dict = torch.load(ckpt_path)
models_state = state_dict["models"]

# Check original values BEFORE modification
print("=== BEFORE ===")
print(f"Background _means[0]: {models_state['Background']['_means'][0]}")
print(f"RigidNodes instances_trans[0,0]: {models_state['RigidNodes']['instances_trans'][0, 0]}")

offset = torch.tensor([1.0, 1.0, 1.0], device="cuda")

# Move world-space translations (not local _means)
for class_name in ["RigidNodes", "DeformableNodes"]:
    if "DeformableNodes" in models_state:
        if "instances_trans" in models_state[class_name]:
            models_state[class_name]["instances_trans"] += offset

bg_means = models_state['Background']['_means']
print(f"Background range: min={bg_means.min(0).values}, max={bg_means.max(0).values}")

# Check values AFTER modification
print("\n=== AFTER ===")
print(f"Background _means[0]: {models_state['Background']['_means'][0]}")
print(f"RigidNodes instances_trans[0,0]: {models_state['RigidNodes']['instances_trans'][0, 0]}")

torch.save(state_dict, "outputs/omnire_experiments/exp1/modified_checkpoint.pth")