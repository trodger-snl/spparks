#!/usr/bin/env python3
"""
Test single file processing with grain boundary visualization
"""

import sys
sys.path.insert(0, '/Users/Tron/spparks')

from batch_analyze_spparks_kam_simple import process_single_file_data, determine_optimal_camera_position

def test_single_file_with_boundaries():
    """Test processing a single file with grain boundary visualization."""
    
    print("=" * 60)
    print("TESTING SINGLE FILE WITH GRAIN BOUNDARIES")
    print("=" * 60)
    
    # Test parameters
    filepath = "/Users/Tron/spparks/examples/ReducedTempAM/MisOrientTest/DumpFiles/dump.additive8.10"
    output_dir = "/Users/Tron/spparks/batch_analysis_output"
    crop_bounds = [90, 110, 90, 110, 2, 4]  # Larger region to see grain boundaries better
    
    print(f"Test file: {filepath}")
    print(f"Output directory: {output_dir}")
    print(f"Crop bounds: {crop_bounds}")
    
    # Determine camera position
    camera_position = determine_optimal_camera_position(filepath)
    print(f"Camera position determined")
    
    # Process the file
    args = (
        filepath, camera_position, output_dir, 
        False,  # enable_method1
        True,   # enable_method2 (IPF)
        True,   # enable_kam
        crop_bounds, 
        False,  # enable_vtk
        False   # enable_vtkhdf
    )
    
    print(f"\nProcessing file with grain boundaries enabled...")
    try:
        result = process_single_file_data(args)
        
        print(f"\n" + "=" * 40)
        print("PROCESSING RESULTS")
        print("=" * 40)
        print(f"Timestep: {result['timestep']}")
        print(f"Atoms processed: {result['n_atoms']}")
        print(f"Load time: {result['load_time']:.2f}s")
        print(f"IPF time: {result['method2_time']:.2f}s") 
        print(f"KAM time: {result['kam_time']:.2f}s")
        
        if result['ipf_image']:
            print(f"IPF image: {result['ipf_image']}")
        
        if result['kam_image']:
            print(f"KAM image: {result['kam_image']}")
            
        print(f"\n✓ Single file processing with grain boundaries completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file_with_boundaries()