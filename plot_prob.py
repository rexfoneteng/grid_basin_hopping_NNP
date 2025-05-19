#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2025-, Huu Trong Phan (phanhuutrong93@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20250513 17:07:31
# ==============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set style for professional appearance
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# Generate x values (energy in kJ/mol)
x = np.linspace(0, 20, 1000)

# Calculate probability values using the given function
y = np.exp(-(x/2625.5) / 0.004)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Plot the function
ax.plot(x, y, linewidth=2.5, color='#3274A1')

# Fill area under the curve
ax.fill_between(x, y, alpha=0.2, color='#3274A1')

# Set labels and title
ax.set_xlabel('Energy (kJ/mol)', fontweight='bold')
ax.set_ylabel('Probability', fontweight='bold')
ax.set_title('Probability Distribution Function', fontweight='bold', pad=15)

# Set axis limits
ax.set_xlim(0, 20)
ax.set_ylim(bottom=0)

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Tight layout for better spacing
plt.tight_layout()

# Save figure with high resolution
plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
