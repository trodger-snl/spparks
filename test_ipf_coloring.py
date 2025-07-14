#!/usr/bin/env python3
"""
Test script for IPF coloring with orix library.
This will help debug the correct API usage for the current orix version.
"""

import numpy as np
import matplotlib.pyplot as plt

def test_ipf_coloring():
    """Test IPF coloring with sample quaternion data."""
    
    print("Testing orix IPF coloring...")
    
    try:
        # Import orix components
        from orix.quaternion import Quaternion
        from orix.quaternion import Orientation
        from orix.vector import Vector3d
        from orix.plot import IPFColorKeyTSL
        from orix.crystal_map import Phase
        
        print("✓ Successfully imported orix components")
        
        # Create sample quaternion data (normalized)
        n_samples = 100
        np.random.seed(42)
        quaternions = np.random.randn(n_samples, 4)
        # Normalize quaternions
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        print(f"✓ Created {n_samples} sample quaternions")
        print(f"  Shape: {quaternions.shape}")
        print(f"  Sample quaternion: {quaternions[0]}")
        
        # Test different phase creation methods
        print("\nTesting Phase creation methods...")
        
        # Method 1: Try with symmetry groups
        try:
            from orix.crystal_map import create_coordinate_arrays
            from orix.symmetry import Oh  # Cubic symmetry for FCC
            phase1 = Phase(name='Gamma', symmetry=Oh)
            print("✓ Method 1: Phase with symmetry object works")
        except Exception as e1:
            print(f"✗ Method 1 failed: {e1}")
            phase1 = None
        
        # Method 2: Try with point group string
        try:
            phase2 = Phase(name='Gamma', point_group='m-3m')
            print("✓ Method 2: Phase with point_group string works")
        except Exception as e2:
            print(f"✗ Method 2 failed: {e2}")
            phase2 = None
            
        # Method 3: Try with space group
        try:
            phase3 = Phase(name='Gamma', space_group=225)  # Fm-3m for FCC
            print("✓ Method 3: Phase with space_group number works")
        except Exception as e3:
            print(f"✗ Method 3 failed: {e3}")
            phase3 = None
        
        # Select working phase
        phase = phase1 or phase2 or phase3
        if phase is None:
            raise ValueError("No phase creation method worked")
        
        print(f"✓ Using phase: {phase}")
        
        # Create quaternion object
        quat_obj = Quaternion(quaternions)
        print(f"✓ Created Quaternion object: {quat_obj.shape}")
        
        # Create orientation object - test different methods
        print("\nTesting Orientation creation...")
        
        # Method 1: Direct with phase
        try:
            ori1 = Orientation(quat_obj, phase=phase)
            print("✓ Orientation with phase parameter works")
            orientation = ori1
        except Exception as e1:
            print(f"✗ Orientation with phase parameter failed: {e1}")
            
            # Method 2: Set phase after creation
            try:
                ori2 = Orientation(quat_obj)
                ori2.phase = phase
                print("✓ Orientation with phase attribute works")
                orientation = ori2
            except Exception as e2:
                print(f"✗ Orientation with phase attribute failed: {e2}")
                raise ValueError("No orientation creation method worked")
        
        print(f"✓ Created Orientation: {orientation}")
        
        # Create IPF direction
        ipf_direction = Vector3d([0, 0, 1])  # Z-direction
        print(f"✓ Created IPF direction: {ipf_direction}")
        
        # Create IPF color key - test different methods
        print("\nTesting IPF color key creation...")
        
        try:
            # Method 1: Use phase.point_group
            ipf_key = IPFColorKeyTSL(phase.point_group)
            print("✓ IPFColorKeyTSL with phase.point_group works")
        except Exception as e:
            print(f"✗ IPFColorKeyTSL with phase.point_group failed: {e}")
            
            try:
                # Method 2: Use symmetry from phase
                ipf_key = IPFColorKeyTSL(phase.symmetry)
                print("✓ IPFColorKeyTSL with phase.symmetry works")
            except Exception as e2:
                print(f"✗ IPFColorKeyTSL with phase.symmetry failed: {e2}")
                
                try:
                    # Method 3: Create symmetry manually
                    from orix.symmetry import get_point_group
                    symmetry = get_point_group('m-3m')
                    ipf_key = IPFColorKeyTSL(symmetry)
                    print("✓ IPFColorKeyTSL with manual symmetry works")
                except Exception as e3:
                    print(f"✗ IPFColorKeyTSL with manual symmetry failed: {e3}")
                    raise ValueError("IPF color key creation failed")
        
        # Generate colors
        print("\nTesting color generation...")
        
        try:
            # Test different API signatures
            try:
                # Method 1: Two arguments
                colors = ipf_key.orientation2color(orientation, ipf_direction)
                print("✓ Method 1: Two arguments works")
            except Exception as e1:
                print(f"✗ Method 1 failed: {e1}")
                
                try:
                    # Method 2: One argument (direction might be set in constructor)
                    colors = ipf_key.orientation2color(orientation)
                    print("✓ Method 2: One argument works")
                except Exception as e2:
                    print(f"✗ Method 2 failed: {e2}")
                    
                    try:
                        # Method 3: Check if direction needs to be set differently
                        ipf_key.direction = ipf_direction
                        colors = ipf_key.orientation2color(orientation)
                        print("✓ Method 3: Set direction as attribute works")
                    except Exception as e3:
                        print(f"✗ Method 3 failed: {e3}")
                        
                        # Method 4: Try with keyword argument
                        try:
                            colors = ipf_key.orientation2color(orientation, direction=ipf_direction)
                            print("✓ Method 4: Keyword argument works")
                        except Exception as e4:
                            print(f"✗ Method 4 failed: {e4}")
                            raise Exception("All color generation methods failed")
            
            print(f"✓ Color generation successful!")
            print(f"  Colors shape: {colors.shape}")
            print(f"  Colors range: [{colors.min():.3f}, {colors.max():.3f}]")
            print(f"  Sample color: {colors[0]}")
            
            # Plot histogram of colors
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            color_names = ['Red', 'Green', 'Blue']
            for i, (ax, color_name) in enumerate(zip(axes, color_names)):
                ax.hist(colors[:, i], bins=20, alpha=0.7, color=color_name.lower())
                ax.set_title(f'{color_name} Channel')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('Count')
            plt.tight_layout()
            plt.savefig('/Users/Tron/spparks/test_ipf_colors.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            return colors, orientation, ipf_key
            
        except Exception as e:
            print(f"✗ Color generation failed: {e}")
            raise
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure orix is installed: pip install orix")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_working_ipf_function(quaternions, crystal_structure='fcc', direction='z'):
    """
    Working IPF color generation function based on successful test.
    
    Args:
        quaternions: Array of quaternions [w, x, y, z]
        crystal_structure: Crystal structure ('fcc', 'bcc', 'hcp')
        direction: IPF reference direction ('x', 'y', 'z', or [x,y,z])
    
    Returns:
        np.array: RGB colors (N, 3) with values in [0, 1]
    """
    from orix.quaternion import Quaternion
    from orix.quaternion import Orientation
    from orix.vector import Vector3d
    from orix.plot import IPFColorKeyTSL
    from orix.crystal_map import Phase
    
    # Create phase based on crystal structure
    if crystal_structure.lower() == 'fcc':
        try:
            # Try different phase creation methods
            from orix.symmetry import Oh
            phase = Phase(name='Gamma', symmetry=Oh)
        except:
            try:
                phase = Phase(name='Gamma', space_group=225)  # Fm-3m
            except:
                phase = Phase(name='Gamma', point_group='m-3m')
    elif crystal_structure.lower() == 'bcc':
        try:
            from orix.symmetry import Oh
            phase = Phase(name='Alpha', symmetry=Oh)
        except:
            try:
                phase = Phase(name='Alpha', space_group=229)  # Im-3m
            except:
                phase = Phase(name='Alpha', point_group='m-3m')
    else:
        raise ValueError(f"Unsupported crystal structure: {crystal_structure}")
    
    # Create quaternion and orientation objects
    quat_obj = Quaternion(quaternions)
    
    # Try different orientation creation methods
    try:
        orientation = Orientation(quat_obj, phase=phase)
    except:
        orientation = Orientation(quat_obj)
        orientation.phase = phase
    
    # Setup IPF direction
    if direction == 'x':
        ipf_dir = Vector3d([1, 0, 0])
    elif direction == 'y':
        ipf_dir = Vector3d([0, 1, 0])
    elif direction == 'z':
        ipf_dir = Vector3d([0, 0, 1])
    else:
        ipf_dir = Vector3d(direction)
    
    # Create IPF color key and generate colors
    try:
        ipf_key = IPFColorKeyTSL(phase.point_group)
    except:
        try:
            ipf_key = IPFColorKeyTSL(phase.symmetry)
        except:
            from orix.symmetry import get_point_group
            symmetry = get_point_group('m-3m')
            ipf_key = IPFColorKeyTSL(symmetry)
    
    colors = ipf_key.orientation2color(orientation, ipf_dir)
    
    return colors

if __name__ == "__main__":
    # Run the test
    result = test_ipf_coloring()
    
    if result is not None:
        print("\n" + "="*50)
        print("SUCCESS: IPF coloring test passed!")
        print("The working function is ready to use.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("FAILED: IPF coloring test failed.")
        print("Check orix installation and version.")
        print("="*50)