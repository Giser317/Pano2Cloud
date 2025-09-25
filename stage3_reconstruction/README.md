# Stage 3: Contour Extraction and Dense Reconstruction

This stage refines building contours and generates dense point clouds.

---

## Pipeline

1. **A-shape extraction**  
   Extract initial building contours from projected point clouds.  
   → See `ashape_extraction.md`

2. **Snake algorithm with curvature**  
   Refine contours using active contour models, preserving sharp corners.  
   → See `snake_contour.py`

3. **Regularization of building coverage**  
   Simplify and regularize contours into standard geometric forms (straight edges, rounded corners).  
   → See `regularization.md`

4. **Dense/meter-level point cloud generation**  
   Use building masks, RGB images, and georeferenced metadata to produce colored 3D point clouds.  
   → See `dense_pointcloud.py`

---

## Example Usage

```bash
# Step 2: refine contour
python snake_contour.py --input boundary.geojson --points pointcloud.csv --output refined.geojson

# Step 4: generate dense point cloud
python dense_pointcloud.py --points_csv cameras.csv --building_geojson buildings.geojson \
  --out_dir outputs/ --img_dir images/ --mask_dir masks/
