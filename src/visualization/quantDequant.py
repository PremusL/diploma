import numpy as np
import matplotlib.pyplot as plt

# --- 1. Settings ---
ROWS = 3
COLS = 4
QUANT_MIN = -127
QUANT_MAX = 127

# --- 2. Data Generation ---
np.random.seed(42)
original_matrix = (np.random.randn(ROWS, COLS)) * 400

# --- 3. Quantization ---
abs_max = np.max(np.abs(original_matrix))
scale = abs_max / QUANT_MAX

quantized_matrix = np.round(original_matrix / scale)
quantized_matrix = np.clip(quantized_matrix, QUANT_MIN, QUANT_MAX).astype(np.int8)

# --- 4. Dequantization ---
dequantized_matrix = quantized_matrix.astype(float) * scale


# --- 5. Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

plt.style.use('default')

def plot_matrix(ax, data, title, color):
    """Helper function to draw a matrix."""
    ax.imshow(np.ones(data.shape) - 0.555, cmap='binary', vmin=0, vmax=1)
    ax.set_title(title, fontsize=16, pad=20, weight='bold') # Increased font size for title
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.1f}" if isinstance(data[i, j], float) else str(data[i, j])
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=14) # Increased font size for numbers

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(color + '15')
    for spine in ax.spines.values():
        spine.set_edgecolor(color + '10')
        spine.set_linewidth(1.5)


plot_matrix(axes[0], original_matrix, "Originalna matrika (FP32)", "#000000")
plot_matrix(axes[1], quantized_matrix, "Kvantizirana matrika (INT8)", "#051565")
plot_matrix(axes[2], dequantized_matrix, "Dequantizirana matrika (FP32)", "#000000")


fig.text(0.325, 0.54, 'Quantize', ha='center', va='center', fontsize=18, color='#d96f02', weight='bold')
fig.text(0.322, 0.46, f"Range:\n[{QUANT_MIN}, {QUANT_MAX}]", ha='center', va='center', fontsize=14, color='gray') # Increased font
fig.text(0.674, 0.54, 'Dequantize', ha='center', va='center', fontsize=18, color='black', weight='bold')

arrow_style = dict(arrowstyle="->,head_width=0.6,head_length=1.0", color="black", lw=2.5)
plt.annotate("",
             xy=(0.38, 0.5), xycoords='figure fraction',
             xytext=(0.271, 0.5), textcoords='figure fraction',
             arrowprops=arrow_style)
plt.annotate("",
             xy=(0.730, 0.5), xycoords='figure fraction',
             xytext=(0.623, 0.5), textcoords='figure fraction',
             arrowprops=arrow_style)


plt.subplots_adjust(left=0.03, right=0.97, top=0.99, bottom=0.01, wspace=0.47)
plt.show()