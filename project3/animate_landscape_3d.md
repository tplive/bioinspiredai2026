# Animate 3D Landscape Usage

This document explains how to use the 3D landscape animation script.

## Script

Path: scripts/animate_landscape_3d.jl

The script builds a rotating 3D visualization of the sampled fitness landscape and highlights local optima.

## Command Format

julia --project=. scripts/animate_landscape_3d.jl <penalized_fitness.csv> <local_optima.csv> <output.mp4> [sample_fraction] [frames] [fps]

## Parameters

1. penalized_fitness.csv
- CSV with columns: row, decimal, bitstring, accuracy, normalized_time, active_features, penalized_fitness.

2. local_optima.csv
- CSV with columns: row, decimal, fitness, bitstring.

3. output.mp4
- Destination video file path.

4. sample_fraction (optional)
- Fraction of states sampled uniformly for plotting.
- Range: (0, 1].
- Default: 0.20.

5. frames (optional)
- Total animation frames.
- Must be >= 2.
- Default: 180.

6. fps (optional)
- Video framerate.
- Must be >= 1.
- Default: 24.

## Examples

Example A (default optional values):

julia --project=. scripts/animate_landscape_3d.jl artifacts/05-nsga-ii/penalized_fitness.csv artifacts/05-nsga-ii/local_optima.csv artifacts/05-nsga-ii/fitness_landscape_3d_rotation.mp4

Example B (custom sampling and smoother rotation):

julia --project=. scripts/animate_landscape_3d.jl artifacts/05-nsga-ii/penalized_fitness.csv artifacts/05-nsga-ii/local_optima.csv artifacts/05-nsga-ii/fitness_landscape_3d_rotation.mp4 0.15 240 30

## Notes

- The animation rotates around the z-axis by sweeping the camera azimuth from 0 to 360 degrees.
- A lower sample_fraction produces a cleaner but less detailed point cloud.
- A higher frames value produces smoother motion but increases render time and output size.
- Local optima are rendered as red diamonds in every frame.
