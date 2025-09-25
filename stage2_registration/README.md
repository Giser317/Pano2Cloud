# Stage 2: Registration

This stage aligns initial point clouds and images through neighbor analysis, clustering, feature matching, and triangulation.

---

## Pipeline

1. **Nearest neighbor analysis**  
   Match points between neighboring clouds with `nearest_neighbors.py`.

2. **Visual clustering**  
   Cluster facade pixels using LAB + LBP features with `cluster_visual.py`.

3. **Joint clustering**  
   Combine spatial coordinates and semantic types using K-Prototypes (`cluster_joint.py`).

4. **Feature extraction**  
   Extract image correspondences using [LightGlue](https://github.com/cvg/LightGlue).

5. **Forward intersection**  
   Compute 3D intersection points of matched rays (`forward_intersection.py`).

---

## Example Usage

```bash
# Step 1: nearest neighbors
python nearest_neighbors.py --input_dir pointclouds/ --output_dir matches/ --max_distance 1.0

# Step 5: triangulation
python forward_intersection.py --input matches.csv --output triangulated.csv
