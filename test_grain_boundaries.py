#!/usr/bin/env python3
"""
Test grain boundary detection functionality
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/Tron/spparks')

from batch_analyze_spparks_kam_simple import (
    parse_spparks_dump, 
    detect_grain_boundaries,
    apply_subvolume_crop
)

def test_grain_boundary_detection():
    """Test grain boundary detection on a single file."""
    filepath = "/Users/Tron/spparks/examples/ReducedTempAM/MisOrientTest/DumpFiles/dump.additive8.10"
    
    print("=" * 60)
    print("TESTING GRAIN BOUNDARY DETECTION")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {filepath}")
    spparks_data = parse_spparks_dump(filepath)
    atoms = spparks_data['atoms']
    
    # Extract coordinates and quaternions
    coords = atoms[['x', 'y', 'z']].values.astype(float)
    quaternions = atoms[['d1', 'd2', 'd3', 'd4']].values
    grain_ids = atoms['i1'].values
    
    print(f"Loaded {len(coords)} atoms")
    print(f"Coordinate ranges:")
    print(f"  X: {coords[:, 0].min():.1f} - {coords[:, 0].max():.1f}")
    print(f"  Y: {coords[:, 1].min():.1f} - {coords[:, 1].max():.1f}")
    print(f"  Z: {coords[:, 2].min():.1f} - {coords[:, 2].max():.1f}")
    
    # Check grain IDs
    unique_grains = np.unique(grain_ids)
    print(f"Number of unique grains: {len(unique_grains)}")
    print(f"Grain ID range: {unique_grains.min()} - {unique_grains.max()}")
    
    # Apply cropping for faster testing
    crop_bounds = [50, 150, 5, 15, 2, 4]
    print(f"\nApplying crop bounds: {crop_bounds}")
    
    # Apply subvolume crop (using dummy colors for compatibility)
    dummy_colors = np.ones((len(coords), 3))
    coords_cropped, colors_cropped, grain_ids_cropped, quaternions_cropped = apply_subvolume_crop(
        coords, dummy_colors, grain_ids, quaternions, crop_bounds
    )
    
    print(f"After cropping: {len(coords_cropped)} atoms")
    
    # Test different threshold values
    thresholds = [3.0, 5.0, 10.0, 15.0]
    
    for threshold in thresholds:
        print(f"\n" + "-" * 40)
        print(f"Testing threshold: {threshold}°")
        print("-" * 40)
        
        try:
            boundary_edges, boundary_coords = detect_grain_boundaries(
                coords_cropped, quaternions_cropped, 
                threshold_deg=threshold
            )
            
            print(f"Found {len(boundary_edges)} boundary edges")
            print(f"Boundary coordinates shape: {boundary_coords.shape if len(boundary_coords) > 0 else 'None'}")
            
            if len(boundary_edges) > 0:
                print(f"First 5 boundary edges:")
                for i, (p1, p2) in enumerate(boundary_edges[:5]):
                    coord1 = coords_cropped[p1]
                    coord2 = coords_cropped[p2]
                    print(f"  Edge {i+1}: ({coord1[0]:.1f}, {coord1[1]:.1f}, {coord1[2]:.1f}) -> ({coord2[0]:.1f}, {coord2[1]:.1f}, {coord2[2]:.1f})")
            
        except Exception as e:
            print(f"Error with threshold {threshold}°: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("GRAIN BOUNDARY DETECTION TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_grain_boundary_detection()