import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# 6. Add text annotations (requires conversion to numpy)
# Note: This part needs to happen outside JAX JIT
def add_text_labels(avg_weights, name=None):
    # 1. Mask for explored cells
    if name is not None and "penalty" in name:
        explored = avg_weights > 0
    elif name is not None and "value" in name:
        explored = np.abs(avg_weights) > 1e-6
    else: # if name is None or "penalty" not in name, no masking
        explored = np.ones_like(avg_weights, dtype=bool) # set all to true

    explored_values = avg_weights[explored]

    # 2. Normalize only over explored values
    if explored_values.size > 0:
        min_val = explored_values.min()
        max_val = explored_values.max()
    else:
        min_val = max_val = 0.0  # fallback if everything is unexplored

    normalized = np.zeros_like(avg_weights)
    if max_val > min_val:
        normalized[explored] = (avg_weights[explored] - min_val) / (max_val - min_val + 1e-8)

    # 3. Colormap: white → blue
    def white_blue(x):
        # x = np.clip(x, 0.0, 1.0)
        # target_color = np.array([0.0, 0.0, 1.0])  # pure blue
        # white = np.ones_like(target_color)
        # return (1 - x[..., None]) * white + x[..., None] * target_color
        x = np.clip(x, 0.1, 1)
        r = 1 - x
        g = 1 - x
        b = np.ones_like(x)
        return np.stack([r, g, b], axis=-1)

    rgb_heatmap = (255 * white_blue(normalized)).astype(np.uint8)

    # 4. Black out unexplored cells
    rgb_heatmap[~explored] = [0, 0, 0]

    # 5. Plot with labels
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_heatmap, interpolation='nearest')
    grid_rows = avg_weights.shape[0]
    grid_cols = avg_weights.shape[1]

    def format_number(val):
        return f"{val:.2f}" if abs(val) >= 0.01 else f"{val:.1e}"

    for i in range(grid_rows):
        for j in range(grid_cols):
            if explored[i, j]:  # skip blacked-out squares
                ax.text(j, i, format_number(avg_weights[i, j]),
                        ha="center", va="center",
                        color="black" if normalized[i, j] < 0.5 else "white",
                        fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # 6. Convert figure to RGB image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[..., :3]  # Drop alpha channel
    plt.close()
    return img

def create_heatmap(row, col, weights, grid_shape=(8,8)):
    # 1. Calculate sum and count for each cell
    grid_rows, grid_cols = grid_shape
    sum_weights, _, _ = jnp.histogram2d(
        row, col,
        bins=[grid_rows, grid_cols],
        weights=weights,
        range=[[0, grid_rows], [0, grid_cols]]
    )
    
    counts, _, _ = jnp.histogram2d(
        row, col,
        bins=[grid_rows, grid_cols],
        range=[[0, grid_rows], [0, grid_cols]]
    )
    
    # 2. Compute averages with safe division
    avg_weights = jnp.where(
        counts > 0,
        sum_weights / (counts + 1e-8),  # +epsilon to avoid NaN
        0.0
    )

    # Return both JAX array and text-annotated version
    return avg_weights