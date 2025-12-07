import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the checkpoint
checkpoint_path = 'outputs/training/run_2025-12-04_11-08-29/checkpoints/checkpoint_ep_52.pt'
ckpt = torch.load(checkpoint_path, map_location='cpu')
history = ckpt['training_history']

# Extract loss data
policy_loss = history['policy_loss']
value_loss = history['value_loss']
total_loss = history['total_loss']

# Create epoch numbers (1-indexed)
epochs = list(range(1, len(policy_loss) + 1))

# Create output directory
output_dir = 'outputs/training/run_2025-12-04_11-08-29'

# 1. Policy Loss plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, policy_loss, 'g-', marker='o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Policy Loss', fontsize=14)
plt.title('Policy Loss over Training Epochs', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(left=min(epochs), right=max(epochs))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'policy_loss.png'), dpi=300, bbox_inches='tight')
print(f"Saved policy loss plot to {os.path.join(output_dir, 'policy_loss.png')}")
plt.close()

# 2. Value Loss plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, value_loss, 'orange', marker='s', linewidth=2, linestyle='--', markersize=8)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Value Loss', fontsize=14)
plt.title('Value Loss over Training Epochs', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(left=min(epochs), right=max(epochs))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'value_loss.png'), dpi=300, bbox_inches='tight')
print(f"Saved value loss plot to {os.path.join(output_dir, 'value_loss.png')}")
plt.close()

# 3. Total Loss plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, total_loss, 'b-', marker='^', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Total Loss', fontsize=14)
plt.title('Total Loss over Training Epochs', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(left=min(epochs), right=max(epochs))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_loss.png'), dpi=300, bbox_inches='tight')
print(f"Saved total loss plot to {os.path.join(output_dir, 'total_loss.png')}")
plt.close()

print("\n" + "="*60)
print("All separate loss plots saved successfully!")
print("="*60)
