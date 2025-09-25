# Stage 1: Sparse Cloud Initialization

This stage prepares initial sparse point clouds from street-view panoramas. It consists of four steps:

1. **Panorama projection & distortion removal**  
   Convert equirectangular panoramas into cubemap faces, correct distortions, and optionally reproject them back.  
   → See `projection.py`

2. **Depth estimation**  
   Apply monocular depth estimation with [ZoeDepth](https://github.com/isl-org/ZoeDepth).  
   → See `depth_estimation.md`

3. **Semantic segmentation**  
   Segment facades and urban objects using [PSPNet](https://github.com/hellochick/PSPNet-tensorflow).  
   → See `segmentation.md`

4. **Initial point cloud generation**  
   Convert depth and mask images into georeferenced 3D point clouds.  
   → See `pointcloud.py`

---

## Example Usage

### Step 4: Point Cloud Generation
```bash
python pointcloud.py \
  --camera_file camera.xlsx \
  --depth_dir depth_images/ \
  --mask_dir masks/ \
  --output_dir outputs/
