import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
color_input = '#E8F4F8'
color_conv = '#B3D9E6'
color_fc = '#7FB3D5'
color_shared = '#4A90A4'
color_output = '#2C5F75'
color_roe = '#F4E8D8'
color_roe_fc = '#D4C4A8'

# Helper function to draw boxes
def draw_box(x, y, w, h, label, sublabel, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
            fontsize=fontsize, weight='bold')
    ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha='center', va='center',
            fontsize=fontsize-1, style='italic')

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
                fontsize=7, style='italic', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor='gray', alpha=0.8))

# Title
ax.text(7, 11.5, 'AlphaZero Policy-Value Network Architecture',
        ha='center', fontsize=14, weight='bold')

# ============= TOP STREAM: Voxel Grid Processing =============
# Input: Voxel Grid
draw_box(0.5, 9, 1.5, 1, 'Belief Grid', '20×20×20', color_input)
ax.text(1.25, 10.2, 'Input', ha='center', fontsize=8, style='italic')

# Conv3D Layer 1
draw_arrow(2, 9.5, 3, 9.5)
draw_box(3, 9, 1.5, 1, 'Conv3D', '16 filters\n3×3×3, s=1', color_conv, fontsize=8)
ax.text(3.75, 10.2, '→ 20×20×20×16', ha='center', fontsize=7)

# Conv3D Layer 2
draw_arrow(4.5, 9.5, 5.5, 9.5)
draw_box(5.5, 9, 1.5, 1, 'Conv3D', '32 filters\n3×3×3, s=2', color_conv, fontsize=8)
ax.text(6.25, 10.2, '→ 10×10×10×32', ha='center', fontsize=7)

# Conv3D Layer 3
draw_arrow(7, 9.5, 8, 9.5)
draw_box(8, 9, 1.5, 1, 'Conv3D', '64 filters\n3×3×3, s=2', color_conv, fontsize=8)
ax.text(8.75, 10.2, '→ 5×5×5×64', ha='center', fontsize=7)

# Flatten + FC
draw_arrow(9.5, 9.5, 10.5, 9.5)
draw_box(10.5, 9, 1.5, 1, 'Flatten + FC', '8000 → 128', color_fc, fontsize=8)
ax.text(11.25, 10.2, 'Grid Features', ha='center', fontsize=7, weight='bold')

# ============= BOTTOM STREAM: ROE Processing =============
# Input: ROE
draw_box(0.5, 6.5, 1.5, 1, 'ROE State', '6D vector', color_roe)
ax.text(1.25, 7.7, 'Input', ha='center', fontsize=8, style='italic')

# ROE Scaling
draw_arrow(2, 7, 3, 7)
draw_box(3, 6.5, 1.5, 1, 'Scale ×10⁴', 'Normalize', color_roe_fc, fontsize=8)

# ROE FC
draw_arrow(4.5, 7, 5.5, 7)
draw_box(5.5, 6.5, 1.5, 1, 'FC + ReLU', '6 → 128', color_roe_fc, fontsize=8)
ax.text(6.25, 7.7, 'ROE Features', ha='center', fontsize=7, weight='bold')

# ============= CONCATENATION =============
# Draw arrows converging to concat
draw_arrow(11.25, 9, 11.25, 8.2)
draw_arrow(6.25, 7.5, 10.5, 8.2)

# Concatenation
draw_box(10.5, 7.7, 1.5, 0.8, 'Concatenate', '256D', color_shared, fontsize=8)

# ============= SHARED LAYERS =============
draw_arrow(11.25, 7.7, 11.25, 6.9)
draw_box(10.5, 6.3, 1.5, 0.8, 'FC + ReLU', '256 → 128', color_shared, fontsize=8)
ax.text(11.25, 7.15, 'Shared\nRepresentation', ha='center', fontsize=7,
        weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))

# ============= OUTPUT HEADS =============
# Policy Head
draw_arrow(11.25, 6.3, 11.25, 5.5)
draw_arrow(11.25, 5.2, 10, 4.5)
draw_box(9, 4, 2, 0.8, 'Policy Head', 'FC: 128 → 13', color_output, fontsize=8)
ax.text(10, 5, 'Softmax', ha='center', fontsize=7, style='italic')

# Policy output
draw_arrow(10, 4, 10, 3.2)
draw_box(9, 2.5, 2, 0.8, 'Policy π(a|s)', '13 actions', '#FFD700', fontsize=9)
ax.text(10, 2.1, 'Action probabilities', ha='center', fontsize=7, style='italic')

# Value Head
draw_arrow(11.25, 6.3, 11.25, 5.5)
draw_arrow(11.25, 5.2, 12.5, 4.5)
draw_box(12, 4, 2, 0.8, 'Value Head', 'FC: 128 → 1', color_output, fontsize=8)

# Value output
draw_arrow(13, 4, 13, 3.2)
draw_box(12, 2.5, 2, 0.8, 'Value V(s)', 'scalar', '#90EE90', fontsize=9)
ax.text(13, 2.1, 'State value estimate', ha='center', fontsize=7, style='italic')

# ============= ANNOTATIONS =============
# Add ReLU annotations
relu_y = 9.3
for x_pos in [3.75, 6.25, 8.75]:
    ax.text(x_pos, relu_y - 0.3, 'ReLU', ha='center', fontsize=6,
            color='red', weight='bold')

ax.text(6.25, 6.8, 'ReLU', ha='center', fontsize=6, color='red', weight='bold')
ax.text(11.25, 6.6, 'ReLU', ha='center', fontsize=6, color='red', weight='bold')

# Add dimension flow annotations
ax.annotate('', xy=(13.5, 9.5), xytext=(13.5, 2.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(13.8, 6, 'Feature\nExtraction\n&\nFusion', ha='left', va='center',
        fontsize=8, color='gray', rotation=90)

# Network info box
info_text = (
    'Network Parameters:\n'
    '• Total params: ~1.1M\n'
    '• 3D Conv params: ~50K\n'
    '• FC params: ~1.05M\n'
    '• Batch norm: None\n'
    '• Dropout: None'
)
ax.text(1, 4.5, info_text, ha='left', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                  edgecolor='black', alpha=0.7))

# Add legend for activation functions
legend_elements = [
    mpatches.Patch(color=color_input, label='Input Layer'),
    mpatches.Patch(color=color_conv, label='3D Convolution'),
    mpatches.Patch(color=color_fc, label='Fully Connected'),
    mpatches.Patch(color=color_shared, label='Shared Layer'),
    mpatches.Patch(color=color_output, label='Output Head'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
          framealpha=0.9, bbox_to_anchor=(0, 0))

# Add architectural notes
notes_text = (
    'Key Features:\n'
    '1. Dual-stream architecture\n'
    '2. 3D spatial reasoning (voxel grid)\n'
    '3. Combined state representation\n'
    '4. Shared backbone for efficiency\n'
    '5. Dual outputs (policy + value)'
)
ax.text(1, 2.5, notes_text, ha='left', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8E8',
                  edgecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Network architecture diagram saved as 'network_architecture.png'")
plt.close()