<img src="data\facade-10.png"></img>

# Input Data–Oriented Optimization of VGGT for Low-Compute Devices

This repository contains our course project for **EE 782: Advanced Topics in Machine Learning** at the **Department of Electrical Engineering, IIT Bombay**.  
The project studies how the Visual Geometry Grounded Transformer (VGGT) can be optimized for low-compute hardware by selecting a minimal yet informative subset of input frames.

## Team Members
- **Abhijat Bharadwaj** – 210020002
- **Parth Bansal** – 21d170028
- **Animesh Kumar** – 21d070012
- **Instructor:** Prof. Amit Sethi  

## Project Description
VGGT is a transformer-based 3D reconstruction model that predicts depth maps, point clouds, and camera poses in a single feed-forward pass. While highly accurate, its large size makes deployment difficult on resource-limited devices such as mobile robots, augmented reality platforms, and embedded vision systems.

This project adopts a **data-centric optimization** approach. Instead of modifying or compressing VGGT, we reduce the number of input frames by selecting only the most informative ones. Four sampling strategies are evaluated:

- Uniform Sampling  
- Random Sampling  
- k-Means Sampling  
- k-Center Sampling  

Experiments on ETH3D-SLAM and ETH3D-Stereo datasets show that **k-Center sampling** provides the strongest performance by selecting diverse and representative frames, achieving low trajectory error while using only a small fraction of the original frames.  
Our results demonstrate that VGGT maintains strong 3D reconstruction and pose estimation quality even with aggressive input reduction.

## Key Features
- Evaluation on multi-view datasets with varying scene lengths  
- Frame sampling using DINO feature embeddings   
- Absolute Trajectory Error (ATE) computation  
- Comparative plots for ATE, trajectories, and point clouds  
- Reproducible scripts requiring only a lightweight GPU  

## Files
- `Best_Subset_Sampling_using_DiNO.ipynb` : (main) Contains details of our work in optimal input subset selection
- `VGGT_quantized.ipynb` : Contains our exploration of post-training quantization of VGGT
- `view-3d-point-cloud.py` : Helper script to visualize 3D point clouds generated from VGGT using open3d
-  `data\einstein_1\*.pt` : point cloud and image files, generated during 3D reconstruction of einstein_1 scene of ETH3D using first 30 frames, uniformly sampled 15 frames, and uniformly sampled 5 frames 