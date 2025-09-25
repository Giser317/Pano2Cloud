# Pano2Cloud
a three-stage framework for reconstructing building facades from street-view panoramas. 
# Pano2Cloud: A Multistage Framework for 3D Building Facade Reconstruction from Street-View Images

This repository contains the reference implementation of **Pano2Cloud**, 
a three-stage framework for reconstructing building facades from street-view panoramas.  
It accompanies our paper:


---
## Motivation

Street-view panoramas are an abundant and low-cost data source for urban 3D modeling.
However, reconstructing building facades from such data is challenging due to:

Distortions in panoramic projection

Uncertainty in monocular depth estimation

Large baseline differences between views

Complex facade occlusions and irregular contours

Pano2Cloud addresses these challenges by decomposing the reconstruction pipeline into three modular stages,
each of which can be run independently or combined into an end-to-end framework.

## Framework Overview

Pano2Cloud follows a **three-stage pipeline**:

1. **Sparse cloud initialization**  
   - Panorama projection & distortion removal  
   - Depth estimation (ZoeDepth)  
   - Semantic segmentation (PSPNet)  
   - Initial point cloud generation  
   → See [`stage1_sparse_cloud/`](stage1_sparse_cloud/)

2. **Registration & clustering**  
   - Pairwise point cloud nearest-neighbor analysis  
   - Visual and joint clustering  
   - Feature matching (LightGlue)  
   - Forward intersection for registered points  
   → See [`stage2_registration/`](stage2_registration/)

3. **Contour extraction & dense reconstruction**  
   - A-shape contour extraction  
   - Snake-based contour refinement with curvature constraints  
   - Regularization of building coverage  
   - Meter-level dense point cloud generation  
   → See [`stage3_contour_reconstruction/`](stage3_contour_reconstruction/)

---

## Repository Structure
Pano2Cloud/
├── stage1_sparse_cloud/            # Sparse cloud initialization
├── stage2_registration/            # Registration & clustering
├── stage3_contour_reconstruction/  # Contour & dense reconstruction
│
├── data/                           # Minimal demo dataset
├── demo.sh                         # One-click demo pipeline
├── requirements.txt                # Dependencies
├── LICENSE                         # License (MIT)
├── CITATION.cff                    # Citation metadata
└── README.md                       # Documentation




## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Giser317/Pano2Cloud.git
cd Pano2Cloud
pip install -r requirements.txt

## IAcknowledgements

This framework builds upon and integrates the following open-source projects:

ZoeDepth
 for monocular depth estimation

PSPNet
 for semantic segmentation

LightGlue
 for feature matching

We sincerely thank their authors for releasing the code.
