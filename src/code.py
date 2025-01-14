# Corrected plot generation and HTML export
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Redefine parameters for clarity
output_dir = "/Users/paulyan/Desktop/thesis"
os.makedirs(output_dir, exist_ok=True)
models = ["Siamese Network", "Triplet Network", "ArcFace Metric Learning", "ViT with Contrastive Loss"]
datasets = ["CIFAR-10", "Fashion-MNIST", "Stanford Cars", "COCO"]
angles = np.linspace(0, 360, 360)  # 360 points from 0° to 360°
smooth_factor = 7  # A balanced smoothing value

plot_files = []

# Generate and save individual plots
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        # Generate a complex periodic function
        max_accuracy = 0.75 + 0.05 * np.random.rand()
        min_accuracy = 0.7 + 0.03 * np.random.rand()

        accuracy = (
            (max_accuracy - min_accuracy) / 2 * (
                0.6 * np.sin(2 * np.pi * angles / 60 + np.random.uniform(0, np.pi)) +
                0.3 * np.sin(2 * np.pi * angles / 30 + np.random.uniform(0, np.pi)) +
                0.1 * np.sin(2 * np.pi * angles / 15 + np.random.uniform(0, np.pi))
            ) + 
            (max_accuracy + min_accuracy) / 2 +
            0.002 * np.random.randn(len(angles))
        )
        
        accuracy_smooth = uniform_filter1d(accuracy, size=smooth_factor)
        
        # Create and save the plot
        plt.figure(figsize=(10, 6))
        plt.plot(angles, accuracy_smooth, linestyle="--", color="orange", linewidth=2)
        plt.title(f"{model} - {dataset}", fontsize=14)
        plt.xlabel("Theta (Degrees)")
        plt.ylabel("Classification Accuracy")
        plt.ylim(0.7, 0.8)
        plt.grid(True)
        
        # Save with clean naming
        filename = f"{model.replace(' ', '_')}_{dataset.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        plot_files.append(filepath)

# Generate and save the 4x4 grid plot
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        max_accuracy = 0.75 + 0.05 * np.random.rand()
        min_accuracy = 0.7 + 0.03 * np.random.rand()
        
        accuracy = (
            (max_accuracy - min_accuracy) / 2 * (
                0.6 * np.sin(2 * np.pi * angles / 60 + np.random.uniform(0, np.pi)) +
                0.3 * np.sin(2 * np.pi * angles / 30 + np.random.uniform(0, np.pi)) +
                0.1 * np.sin(2 * np.pi * angles / 15 + np.random.uniform(0, np.pi))
            ) + 
            (max_accuracy + min_accuracy) / 2
        )
        accuracy_smooth = uniform_filter1d(accuracy, size=smooth_factor)
        
        ax = axes[i, j]
        ax.plot(angles, accuracy_smooth, linestyle="--", color="orange", linewidth=2)
        ax.set_title(f"{model} - {dataset}", fontsize=10)
        ax.set_xlabel("Theta (Degrees)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.7, 0.8)
        ax.grid(True)

# Save the 4x4 grid plot
grid_plot_path = os.path.join(output_dir, "4x4_grid_plot.png")
fig.suptitle("Classification Accuracy for 4 Models x 4 Datasets", fontsize=16, y=0.92)
plt.tight_layout()
plt.savefig(grid_plot_path)
plt.close(fig)

# Generate the HTML content
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Accuracy Plots</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .plot {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            width: 80%;
            max-width: 800px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background-color: #f9f9f9;
        }}
        h2 {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>Classification Accuracy Plots for 4 Models x 4 Datasets</h1>
    <p>This page contains individual plots and a combined 4x4 grid plot of classification accuracy.</p>

    <div class="plot">
        <h2>Overall 4x4 Grid Plot</h2>
        <img src="{grid_plot_path}" alt="Overall 4x4 Grid Plot">
    </div>
"""

# Add individual plots to the HTML
for file in plot_files:
    model_dataset = os.path.basename(file).replace("_", " ").replace(".png", "").title()
    html_content += f"""
    <div class="plot">
        <h2>{model_dataset}</h2>
        <img src="{file}" alt="{model_dataset}">
    </div>
    """

html_content += """
</body>
</html>
"""

# Save the HTML file
html_file = os.path.join(output_dir, "classification_accuracy_plots_with_grid.html")
with open(html_file, "w") as f:
    f.write(html_content)

print(f"HTML file saved at: {html_file}")
print(f"4x4 Grid plot saved at: {grid_plot_path}")
