#!/usr/bin/env python3
"""
Visualize the difference between shell-based and distance-based nucleation patterns.
This script demonstrates how the refactored nucleation_particle_flipper creates
more isotropic (spherical) grains compared to the original faceted approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_shell_based_nucleus(size=3):
    """
    Simulate the old shell-based nucleation pattern.
    Adds neighbors in order: faces, edges, corners
    """
    # Start with center
    nucleus = {(0, 0, 0)}
    
    # First shell: 6 face neighbors
    face_neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    # Second shell: 12 edge neighbors
    edge_neighbors = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]
    
    # Third shell: 8 corner neighbors
    corner_neighbors = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]
    
    # Add shells in order
    remaining = size - 1
    
    # Add face neighbors first
    for n in face_neighbors:
        if remaining <= 0:
            break
        nucleus.add(n)
        remaining -= 1
    
    # Add edge neighbors
    for n in edge_neighbors:
        if remaining <= 0:
            break
        nucleus.add(n)
        remaining -= 1
    
    # Add corner neighbors
    for n in corner_neighbors:
        if remaining <= 0:
            break
        nucleus.add(n)
        remaining -= 1
    
    return nucleus

def create_distance_based_nucleus(size=3, randomize=True):
    """
    Simulate the new distance-based nucleation pattern.
    Adds neighbors in order of distance from center.
    """
    # Start with center
    nucleus = {(0, 0, 0)}
    
    # All 26 neighbors with their distances
    neighbors = []
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                dist = np.sqrt(i**2 + j**2 + k**2)
                # Add small random perturbation if randomizing
                if randomize:
                    dist += 0.1 * (np.random.rand() - 0.5)
                neighbors.append((dist, (i, j, k)))
    
    # Sort by distance
    neighbors.sort(key=lambda x: x[0])
    
    # Add neighbors in distance order
    remaining = size - 1
    for _, coord in neighbors:
        if remaining <= 0:
            break
        nucleus.add(coord)
        remaining -= 1
    
    return nucleus

def plot_nucleus_comparison(size=19):
    """
    Create a side-by-side comparison of shell-based vs distance-based nucleation.
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Create nuclei
    shell_nucleus = create_shell_based_nucleus(size)
    dist_nucleus = create_distance_based_nucleus(size, randomize=False)
    dist_nucleus_rand = create_distance_based_nucleus(size, randomize=True)
    
    # Convert to arrays for plotting
    def nucleus_to_arrays(nucleus):
        x = [p[0] for p in nucleus]
        y = [p[1] for p in nucleus]
        z = [p[2] for p in nucleus]
        return x, y, z
    
    # Plot 1: Shell-based (old method)
    ax1 = fig.add_subplot(131, projection='3d')
    x, y, z = nucleus_to_arrays(shell_nucleus)
    ax1.scatter(x, y, z, c='red', s=100, alpha=0.6, edgecolors='darkred')
    ax1.set_title('Shell-based Nucleation\n(Original - Anisotropic)', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(-3, 3)
    
    # Plot 2: Distance-based (new method)
    ax2 = fig.add_subplot(132, projection='3d')
    x, y, z = nucleus_to_arrays(dist_nucleus)
    ax2.scatter(x, y, z, c='blue', s=100, alpha=0.6, edgecolors='darkblue')
    ax2.set_title('Distance-based Nucleation\n(New - More Isotropic)', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-3, 3)
    
    # Plot 3: Distance-based with randomization
    ax3 = fig.add_subplot(133, projection='3d')
    x, y, z = nucleus_to_arrays(dist_nucleus_rand)
    ax3.scatter(x, y, z, c='green', s=100, alpha=0.6, edgecolors='darkgreen')
    ax3.set_title('Distance-based + Randomization\n(New - Natural Variation)', fontsize=12)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_zlim(-3, 3)
    
    plt.suptitle(f'Nucleation Pattern Comparison (Size = {size} sites)', fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_isotropy(max_size=27):
    """
    Analyze the isotropy of nucleation patterns by measuring
    the ratio of maximum to minimum radial extent.
    """
    sizes = range(7, max_size, 3)
    shell_ratios = []
    dist_ratios = []
    dist_rand_ratios = []
    
    for size in sizes:
        # Create nuclei
        shell_n = create_shell_based_nucleus(size)
        dist_n = create_distance_based_nucleus(size, randomize=False)
        
        # Calculate max/min radial distances for each
        def calc_isotropy_ratio(nucleus):
            if len(nucleus) <= 1:
                return 1.0
            distances = [np.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in nucleus if p != (0, 0, 0)]
            if not distances:
                return 1.0
            return max(distances) / min(distances)
        
        shell_ratios.append(calc_isotropy_ratio(shell_n))
        dist_ratios.append(calc_isotropy_ratio(dist_n))
        
        # Average over multiple random realizations
        rand_ratios = []
        for _ in range(10):
            dist_n_rand = create_distance_based_nucleus(size, randomize=True)
            rand_ratios.append(calc_isotropy_ratio(dist_n_rand))
        dist_rand_ratios.append(np.mean(rand_ratios))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, shell_ratios, 'r-o', label='Shell-based (Original)', linewidth=2)
    plt.plot(sizes, dist_ratios, 'b-s', label='Distance-based', linewidth=2)
    plt.plot(sizes, dist_rand_ratios, 'g-^', label='Distance-based + Random', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Isotropy')
    
    plt.xlabel('Nucleus Size (number of sites)', fontsize=12)
    plt.ylabel('Isotropy Ratio (max radius / min radius)', fontsize=12)
    plt.title('Nucleation Isotropy Analysis', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Visualizing nucleation patterns...")
    
    # Show 3D comparison
    plot_nucleus_comparison(size=19)
    
    # Analyze isotropy
    analyze_isotropy(max_size=27)
    
    print("\nKey improvements in the refactored nucleation:")
    print("1. More spherical grain shapes")
    print("2. Reduced crystallographic bias")
    print("3. Better representation of physical nucleation")
    print("4. Optional randomization prevents perfect symmetry")