---
layout: default
---

![Single-Shot Cuboids](./assets/images/graphical_abstract.png "Single-Shot Cuboids")

# Abstract

> It has been shown that global scene understanding tasks like layout estimation can benefit from wider field of views, and specifically spherical panoramas. While much progress has been made recently, all previous approaches rely on intermediate representations and postprocessing to produce Manhattan-aligned estimates. In this work we show how to estimate full room layouts in a single-shot, eliminating the need for postprocessing. Our work is the first to directly infer Manhattan-aligned outputs. To achieve this, our data-driven model exploits direct coordinate regression and is supervised end-to-end. As a result, we can explicitly add quasi-Manhattan constraints, which set the necessary conditions for a homography-based Manhattan alignment module. Finally, we introduce the geodesic heatmaps and loss and a boundary-aware center of mass calculation that facilitate higher quality keypoint estimation in the spherical domain.
Our models and code are publicly available at [https://github.com/VCL3D/SingleShotCuboids](https://github.com/VCL3D/SingleShotCuboids).

# Highlights

- **Single-shot**, end-to-end, spherical panorama-based cuboid layout corner estimation.

- **Spherical Center of Mass** for boundary-aware keypoint estimation on the sphere.

- **Explicit layout constraints** via direct keypoint estimation.

- **Geodesic Distance Loss** for boundary-aware keypoint estimation on the sphere.

-  **Geodesic Gaussian Heatmap Reconstruction** for spherical center of mass keypoints:

- **Homography-based Cuboid Fitting** that ensures end-to-end full Manhattan alignment.

# Results

## Sun360

<img width=45% src="./assets/images/sun360_1.jpg"><img width=40% src="./assets/images/sun360_1.gif">

<img width=45% src="../assets/images/Sun360_2.jpg"><img width=40% src="../assets/images/sun360_2.gif">

<img width=45% src="assets/images/Sun360_3.jpg"><img width=40% src="assets/images/sun360_3.gif">

<img width=45% src="../../assets/images/Sun360_4.jpg"><img width=40% src="../../assets/images/sun360_4.gif">

## Stanford2D3D

<img width=45% src="./assets/images/s2d3d_1.jpg"/>
<img width=40% src="./assets/images/s2d3d_1.gif"/>

<img width=45% src="./assets/images/s2d3d_2.jpg"/>
<img width=40% src="./assets/images/s2d3d_2.gif"/>

<img width=45% src="./assets/images/s2d3d_3.jpg"/>
<img width=40% src="./assets/images/s2d3d_3.gif"/>

<img width=45% src="./assets/images/s2d3d_4.jpg"/>
<img width=40% src="./assets/images/s2d3d_4.gif"/>

## Structure3D

<img width=45% src="./assets/images/s3d_1.jpg"/>
<img width=40% src="./assets/images/s3d_1.gif"/>

<img width=45% src="./assets/images/s3d_2.jpg"/>
<img width=40% src="./assets/images/s3d_2.gif"/>

<img width=45% src="./assets/images/s3d_3.jpg"/>
<img width=40% src="./assets/images/s3d_3.gif"/>

<img width=45% src="./assets/images/s3d_4.jpg"/>
<img width=40% src="./assets/images/s3d_4.gif"/>

## Kujiale

<img width=45% src="./assets/images/kuj_1.jpg"/>
<img width=40% src="./assets/images/kuj_1.gif"/>

<img width=45% src="./assets/images/kuj_2.jpg"/>
<img width=40% src="./assets/images/kuj_2.gif"/>

<img width=45% src="./assets/images/kuj_3.jpg"/>
<img width=40% src="./assets/images/kuj_3.gif"/>

<img width=45% src="./assets/images/kuj_4.jpg"/>
<img width=40% src="./assets/images/kuj_4.gif"/>
