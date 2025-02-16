import matplotlib.pyplot as plt
import numpy as np

# Set the style to a built-in one and customize
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#F8F9F9'
plt.rcParams['font.family'] = 'sans-serif'

# Data
openmp_threads = [0, 2, 4, 8, 16]
exec_time = [9.686826, 7.897963, 7.617025, 7.612045, 7.607177]

# Create the plot with a square aspect ratio
fig, ax = plt.subplots(figsize=(8, 8))

# Create gradient colors for a more modern look
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(openmp_threads)))

# Main plot
line = ax.plot(openmp_threads, exec_time, '-', color='#1f77b4', linewidth=3, zorder=2)
scatter = ax.scatter(openmp_threads, exec_time, c=colors, s=150, zorder=3, 
                    marker='o', edgecolor='white', linewidth=2)

# Customize the plot
ax.set_xlabel('Number of OpenMP Threads per MPI Process', 
            fontsize=12, labelpad=10, color='#34495E')
ax.set_ylabel('Execution Time (seconds)', 
            fontsize=12, labelpad=10, color='#34495E')

# Customize grid
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Set x-axis ticks and limits
ax.set_xticks(openmp_threads)
ax.set_xlim(-0.5, 16.5)

# Add value labels on top of each point with improved styling and larger font
for x, y in zip(openmp_threads, exec_time):
    ax.text(x, y + 0.1, f'{y:.2f}s', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#2C3E50')

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#CCCCCC')
    spine.set_linewidth(1)

# Add subtle shading for depth
ax.fill_between(openmp_threads, exec_time, 
                min(exec_time) - 0.5, 
                alpha=0.1, color='#1f77b4')

# Adjust layout
plt.tight_layout()

# Save the plot with high quality
plt.savefig('performance_plot.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()