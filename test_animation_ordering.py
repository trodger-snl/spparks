#!/usr/bin/env python3
"""
Test animation creation with proper timestep ordering using existing images
"""

import os
import glob
import cv2
import re
from pathlib import Path

def extract_timestep_from_filename(filename):
    """Extract timestep number from filename."""
    # Extract number after 'timestep_'
    match = re.search(r'timestep_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_animation_sorted(image_pattern, output_path, fps=5):
    """Create MP4 animation from image files with proper sorting."""
    # Get all matching image files
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"No images found matching pattern: {image_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Sort by timestep number
    image_files_sorted = sorted(image_files, key=extract_timestep_from_filename)
    
    print("First 10 files in order:")
    for i, img_path in enumerate(image_files_sorted[:10]):
        timestep = extract_timestep_from_filename(os.path.basename(img_path))
        print(f"  {i+1}: {os.path.basename(img_path)} (timestep {timestep})")
    
    if len(image_files_sorted) > 10:
        print("...")
        print("Last 5 files in order:")
        for i, img_path in enumerate(image_files_sorted[-5:], len(image_files_sorted)-5):
            timestep = extract_timestep_from_filename(os.path.basename(img_path))
            print(f"  {i+1}: {os.path.basename(img_path)} (timestep {timestep})")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files_sorted[0])
    if first_img is None:
        print(f"Error: Could not read first image {image_files_sorted[0]}")
        return
        
    height, width, _ = first_img.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add images to video
    print(f"Creating animation: {output_path}")
    for i, img_path in enumerate(image_files_sorted):
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read image {img_path}")
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(image_files_sorted)} frames")
    
    # Release everything
    out.release()
    print(f"Animation saved: {output_path}")

def test_animation_creation():
    """Test animation creation with existing images."""
    output_dir = "/Users/Tron/spparks/batch_analysis_output"
    
    print("=" * 60)
    print("TESTING ANIMATION CREATION WITH PROPER ORDERING")
    print("=" * 60)
    
    # Test IPF animation
    print("\n1. Testing IPF Animation:")
    ipf_pattern = os.path.join(output_dir, "ipf_timestep_*.png")
    ipf_output = os.path.join(output_dir, "ipf_animation_sorted.mp4")
    create_animation_sorted(ipf_pattern, ipf_output, fps=5)
    
    # Test KAM animation
    print("\n2. Testing KAM Animation:")
    kam_pattern = os.path.join(output_dir, "kam_timestep_*.png")
    kam_output = os.path.join(output_dir, "kam_animation_sorted.mp4")
    create_animation_sorted(kam_pattern, kam_output, fps=5)
    
    # Test Method2 animation (if exists)
    print("\n3. Testing Method2 Animation:")
    method2_pattern = os.path.join(output_dir, "method2_timestep_*.png")
    method2_output = os.path.join(output_dir, "method2_animation_sorted.mp4")
    create_animation_sorted(method2_pattern, method2_output, fps=5)
    
    print("\n" + "=" * 60)
    print("ANIMATION TESTING COMPLETE")
    print("=" * 60)
    print("Created sorted animations:")
    print(f"  - {ipf_output}")
    print(f"  - {kam_output}")
    print(f"  - {method2_output}")

if __name__ == "__main__":
    test_animation_creation()