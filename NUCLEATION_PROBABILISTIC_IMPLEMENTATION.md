# Probabilistic Nucleation with Center-of-Mass Preservation

## Overview

This document describes an implementation approach for improving the isotropy of grain nucleation in SPPARKS while maintaining the grain's center of mass at the original nucleation site.

## Current Issues

1. **Shell-based growth creates anisotropy**: The current implementation adds neighbors in fixed order (faces → edges → corners), creating faceted grains
2. **Recursive growth shifts center**: When a site's neighbors are exhausted, recursion shifts to a new center, creating elongated grains
3. **Cubic lattice bias**: The underlying cubic lattice inherently favors certain growth directions

## Proposed Solution

### Key Concepts

1. **Maintain origin coordinates** throughout recursive calls
2. **Probabilistic selection** based on distance from origin (not current site)
3. **Preserve recursive structure** for simplicity

### Implementation Approach

#### 1. Modified Function Signature

```cpp
void nucleation_particle_flipper(int i, int partRad, RandomPark *random, 
                                double origin_x = -1, double origin_y = -1, double origin_z = -1)
```

- Add optional origin coordinates
- Default values (-1) indicate first call
- Origin is set once and passed through all recursive calls

#### 2. Origin Tracking

```cpp
// First call: record the origin
if (origin_x < 0) {
    origin_x = xyz[i][0];
    origin_y = xyz[i][1];
    origin_z = xyz[i][2];
}
```

#### 3. Distance-Based Probability Calculation

Instead of fixed shell order, calculate probability for each neighbor based on its distance from the **origin**:

```cpp
// For each liquid neighbor j of current site i
double neighbor_x = xyz[neighbor[i][j]][0];
double neighbor_y = xyz[neighbor[i][j]][1];
double neighbor_z = xyz[neighbor[i][j]][2];

// Distance from ORIGIN, not from current site
double dist = sqrt(pow(neighbor_x - origin_x, 2) + 
                  pow(neighbor_y - origin_y, 2) + 
                  pow(neighbor_z - origin_z, 2));

// Probability inversely proportional to distance
double prob = 1.0 / (1.0 + shape_factor * dist);
```

#### 4. Weighted Random Selection

```cpp
// Build list of liquid neighbors with their probabilities
struct NeighborCandidate {
    int index;
    double probability;
    double cumulative_prob;
};

vector<NeighborCandidate> candidates;
double total_prob = 0.0;

// Collect all liquid neighbors
for (int j = 0; j < numneigh[i]; j++) {
    if (active_flag[neighbor[i][j]] == 2) { // liquid
        double dist = calculate_dist_from_origin(neighbor[i][j], origin_x, origin_y, origin_z);
        double prob = 1.0 / (1.0 + shape_factor * dist);
        total_prob += prob;
        candidates.push_back({j, prob, total_prob});
    }
}

// Weighted random selection
double rand_val = random->uniform() * total_prob;
for (const auto& cand : candidates) {
    if (rand_val <= cand.cumulative_prob) {
        // Select this neighbor
        int i_chosen = neighbor[i][cand.index];
        flip_site(i_chosen, s_in);
        // ... rest of solidification logic
        break;
    }
}
```

#### 5. Recursive Call with Origin

```cpp
// When recursing, pass the same origin coordinates
nucleation_particle_flipper(neighbor[i][possible_neigh[neighran]], nSites, random,
                           origin_x, origin_y, origin_z);
```

### Configuration Parameters

Add user-configurable parameters:

```cpp
// In header file
double nucleation_shape_factor;  // Controls isotropy (lower = more spherical)

// In input parsing
else if (strcmp(command,"nucleation_shape") == 0) {
    nucleation_shape_factor = atof(arg[0]);
    // Suggested range: 0.0 (spherical) to 1.0 (more cubic)
}
```

### Algorithm Flow

1. **Initial call**: Site i is the nucleation center, record its coordinates
2. **For each neighbor**: Calculate probability based on distance from origin
3. **Weighted selection**: Choose neighbors probabilistically (closer = more likely)
4. **Continue growth**: Add selected neighbors to grain
5. **If more sites needed**: Recursively call on a grain boundary site, **passing origin**
6. **Result**: Approximately spherical growth centered on original site

### Benefits

- **Reduced anisotropy**: Probabilistic selection breaks cubic lattice bias
- **Center preservation**: All growth decisions based on original center
- **Tunable shape**: Shape factor allows control over sphericity
- **Simple implementation**: Minimal changes to existing recursive structure
- **Physical realism**: Mimics actual solidification (growth rate ∝ 1/distance)

### Considerations

1. **Performance**: Slightly more computation per neighbor (distance calculation)
2. **Randomness**: Results vary between runs (physically realistic)
3. **Parameter tuning**: Shape factor needs calibration for specific materials

### Testing Strategy

1. Visualize grain shapes with different shape factors
2. Measure isotropy ratio (max radius / min radius)
3. Compare with experimental grain morphologies
4. Verify center of mass remains at origin

### Future Enhancements

- **Anisotropic shape factors**: Different factors for different directions
- **Temperature-dependent shape**: Link shape factor to cooling rate
- **Crystal orientation effects**: Bias growth along certain crystallographic directions