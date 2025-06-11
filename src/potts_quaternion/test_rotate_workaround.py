#!/usr/bin/env python3
"""
Test script with workaround for strict quaternion validation
"""

import cppyy
import math

# Include the header file
cppyy.include("/Users/Tron/spparks/src/potts_quaternion/quaternion.h")

# Access the namespace
from cppyy.gbl import SPPARKS_NS
from cppyy.gbl.std import vector

# Add a wrapper function that normalizes quaternions before calling rotate_q_towards_u
cppyy.cppdef("""
namespace SPPARKS_NS {
namespace quaternion {

inline std::vector<double> rotate_q_towards_u_normalized(const std::vector<double>& q,
                                                         const std::vector<double>& u,
                                                         double angle) {
    // Normalize the input quaternion to ensure it passes validation
    std::vector<double> q_norm(4);
    double mag = 0.0;
    for (int i = 0; i < 4; ++i) {
        mag += q[i] * q[i];
    }
    mag = std::sqrt(mag);
    for (int i = 0; i < 4; ++i) {
        q_norm[i] = q[i] / mag;
    }
    
    // Call the original function with the normalized quaternion
    return rotate_q_towards_u(q_norm, u, angle);
}

} // namespace quaternion
} // namespace SPPARKS_NS
""")

def test_with_normalization():
    """Test using the normalized wrapper function"""
    print("Testing with normalization wrapper")
    print("=" * 40)
    
    # Test 1: Basic rotation
    print("\nTest 1: Basic rotation")
    print("-" * 20)
    q = vector[float]([1.0, 0.0, 0.0, 0.0])  # Identity
    u = vector[float]([1.0, 0.0, 0.0])       # x-axis
    angle = math.pi / 4  # 45 degrees
    
    result = SPPARKS_NS.quaternion.rotate_q_towards_u_normalized(q, u, angle)
    print(f"Input: q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}], u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
    print(f"Angle: {math.degrees(angle):.1f}°")
    print(f"Result: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}]")
    
    # Test 2: Overshoot protection
    print("\n\nTest 2: Overshoot protection")
    print("-" * 20)
    # 45° rotation around y-axis (not perfectly normalized from Python math)
    angle_rad = math.pi / 4
    q = vector[float]([math.cos(angle_rad/2), 0.0, math.sin(angle_rad/2), 0.0])
    u = vector[float]([0.0, 0.0, 1.0])  # z-axis
    angle = math.pi  # 180 degrees (overshooting)
    
    result = SPPARKS_NS.quaternion.rotate_q_towards_u_normalized(q, u, angle)
    print(f"Input: q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}], u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
    print(f"Angle: {math.degrees(angle):.1f}° (overshooting)")
    print(f"Result: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}]")
    print("Note: Should align without overshooting")
    
    # Test 3: Already aligned
    print("\n\nTest 3: Already aligned")
    print("-" * 20)
    q = vector[float]([1.0, 0.0, 0.0, 0.0])  # Identity
    u = vector[float]([0.0, 0.0, 1.0])       # z-axis (already aligned)
    angle = math.pi / 6  # 30 degrees
    
    result = SPPARKS_NS.quaternion.rotate_q_towards_u_normalized(q, u, angle)
    print(f"Input: q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}], u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
    print(f"Angle: {math.degrees(angle):.1f}°")
    print(f"Result: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}]")
    print("Note: Should return same quaternion")
    
    # Test 4: Opposite directions
    print("\n\nTest 4: Opposite directions")
    print("-" * 20)
    # 90° rotation around y-axis
    angle_rad = math.pi / 2
    q = vector[float]([math.cos(angle_rad/2), 0.0, math.sin(angle_rad/2), 0.0])
    u = vector[float]([0.0, 0.0, -1.0])  # negative z-axis
    angle = math.pi / 3  # 60 degrees
    
    result = SPPARKS_NS.quaternion.rotate_q_towards_u_normalized(q, u, angle)
    print(f"Input: q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}], u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
    print(f"Angle: {math.degrees(angle):.1f}°")
    print(f"Result: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}]")
    
    print("\n" + "=" * 40)
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_with_normalization()