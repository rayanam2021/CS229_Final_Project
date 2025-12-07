import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the checkpoint
checkpoint_path = 'output_training/run_2025-12-04_11-08-29/checkpoints/checkpoint_ep_52.pt'
ckpt = torch.load(checkpoint_path, map_location='cpu')
history = ckpt['training_history']

# Extract loss data
policy_loss = history['policy_loss']
value_loss = history['value_loss']
total_loss = history['total_loss']

# Create epoch numbers (1-indexed)
epochs = list(range(1, len(policy_loss) + 1))

# Create output directory
output_dir = 'output_training/run_2025-12-04_11-08-29'

# Create individual plots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Policy Loss
axes[0].plot(epochs, policy_loss, 'g-', marker='o', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Policy Loss', fontsize=12)
axes[0].set_title('Policy Loss over Training Epochs', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(left=min(epochs), right=max(epochs))

# Value Loss
axes[1].plot(epochs, value_loss, 'orange', marker='s', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Value Loss', fontsize=12)
axes[1].set_title('Value Loss over Training Epochs', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(left=min(epochs), right=max(epochs))

# Total Loss
axes[2].plot(epochs, total_loss, 'b-', marker='^', linewidth=2)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Total Loss', fontsize=12)
axes[2].set_title('Total Loss over Training Epochs', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(left=min(epochs), right=max(epochs))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_plots_individual.png'), dpi=300, bbox_inches='tight')
print(f"Saved individual loss plots to {os.path.join(output_dir, 'loss_plots_individual.png')}")
plt.close()

# Create combined plot
plt.figure(figsize=(12, 7))
plt.plot(epochs, total_loss, 'b-', marker='^', linewidth=2, label='Total Loss', markersize=8)
plt.plot(epochs, value_loss, 'orange', marker='s', linewidth=2, linestyle='--', label='Value Loss', markersize=8)
plt.plot(epochs, policy_loss, 'g-', marker='o', linewidth=2, label='Policy Loss', markersize=8)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Losses over Epochs', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(left=min(epochs), right=max(epochs))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_plots_combined.png'), dpi=300, bbox_inches='tight')
print(f"Saved combined loss plot to {os.path.join(output_dir, 'loss_plots_combined.png')}")
plt.close()

# Print statistics
print("\n" + "="*60)
print("LOSS STATISTICS")
print("="*60)
print(f"\nPolicy Loss:")
print(f"  Initial: {policy_loss[0]:.6f}")
print(f"  Final:   {policy_loss[-1]:.6f}")
print(f"  Min:     {min(policy_loss):.6f} (Epoch {epochs[policy_loss.index(min(policy_loss))]})")
print(f"  Max:     {max(policy_loss):.6f} (Epoch {epochs[policy_loss.index(max(policy_loss))]})")
print(f"  Mean:    {np.mean(policy_loss):.6f}")
print(f"  Std:     {np.std(policy_loss):.6f}")

print(f"\nValue Loss:")
print(f"  Initial: {value_loss[0]:.6f}")
print(f"  Final:   {value_loss[-1]:.6f}")
print(f"  Min:     {min(value_loss):.6f} (Epoch {epochs[value_loss.index(min(value_loss))]})")
print(f"  Max:     {max(value_loss):.6f} (Epoch {epochs[value_loss.index(max(value_loss))]})")
print(f"  Mean:    {np.mean(value_loss):.6f}")
print(f"  Std:     {np.std(value_loss):.6f}")

print(f"\nTotal Loss:")
print(f"  Initial: {total_loss[0]:.6f}")
print(f"  Final:   {total_loss[-1]:.6f}")
print(f"  Min:     {min(total_loss):.6f} (Epoch {epochs[total_loss.index(min(total_loss))]})")
print(f"  Max:     {max(total_loss):.6f} (Epoch {epochs[total_loss.index(max(total_loss))]})")
print(f"  Mean:    {np.mean(total_loss):.6f}")
print(f"  Std:     {np.std(total_loss):.6f}")

print("\n" + "="*60)
