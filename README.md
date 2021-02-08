# Single-Shot Cuboids: Geodesics-based End-to-end Manhattan Aligned Layout Estimation from Spherical Panoramas

[![Paper](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/pdf/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/SingleShotCuboids/)

This repository contains the code and models for the paper _"Single-Shot Cuboids: Geodesics-based End-to-end Manhattan Aligned Layout Estimation from Spherical Panoramas"_.

> Parts of the code can be used for keypoint localisation using spherical panorama inputs (in equirectangular domain).

## TODO

* [x] Upload Code
* [ ] Upload Model
* [ ] Upload Inference Code

## Code

## Geodesic Distace
The [`GeodesicDistance`](https://github.com/VCL3D/SingleShotCuboids/blob/584aec312fb381b0a02acd89dd2e299f3fdc7ec5/ssc/geodesic_distance.py#L24) module found in [`./ssc/geodesic_distance.py`](https://github.com/VCL3D/SingleShotCuboids/blob/master/ssc/geodesic_distance.py) calculates the great circle or harvesine distance of two coordinates on the sphere. The following image shows the harvesine distance and the corresponding great circle path between points on the equirectangular domain. Distances from the red square to the colored diamonds are also reported in the corresponding color.

```py
                        loss = GeodesicDistance()
```

<p align="center">
  <img src=./assets/images/geodesic.png width=300/>
</p>

An interactive comparison between the geodesic distance and the L2 distance can be run with:

```bash
                        python ssc/geodesic_distance.py
```

Left clicking selects the first (left hand side) point, and right clicking the corresponding second (right hand side) point.
Upon having selected a left and right point, their geodesic and L2 distance will be printed.

## Geodesic Heatmaps
The [`GeodesicGaussian`](https://github.com/VCL3D/SingleShotCuboids/blob/584aec312fb381b0a02acd89dd2e299f3fdc7ec5/ssc/geodesic_gaussian.py#L41) module found in [`./ssc/geodesic_gaussian.py`](https://github.com/VCL3D/SingleShotCuboids/blob/master/ssc/geodesic_gaussian.py) relies on the geodesic distance and reconstructs a Gaussian distribution directly on the equirectangular domain that respects the continuity around the horizontal boundary, and, at the same time, is aware of the equirectangular projection's distortion.

```py
                  module = GeodesicGaussian(std=9.0, normalize=True)
```

The following images show a Gaussian distribution defined on the sphere (left) and the corresponding distribution reconstructed on the equirectangular domain (right).

<img width=15% src="./assets/images/0_sphere.png"><img width=30% src="./assets/images/0_equi.png"><img width=15% src="./assets/images/1_sphere.png"><img width=30% src="./assets/images/1_equi.png">

<img width=15% src="./assets/images/2_sphere.png"><img width=30% src="./assets/images/2_equi.png"><img width=15% src="./assets/images/3_sphere.png"><img width=30% src="./assets/images/3_equi.png">

<img width=15% src="./assets/images/4_sphere.png"><img width=30% src="./assets/images/4_equi.png">

Different (20) random centroid distributions can be visualized by runningwith:

```bash
                python ssc/geodesic_gaussian.py {std: float=9.0} {width: int=512}
```

with the (optional) std argument given in degrees (default: `9.0`), and the (optional) width argument defining the equirectangular pixels at the longitudinal angular coordinate (default: `512`).

## Quasi-Manhattan Center of Mass
The [`QuasiManhattanCenterOfMass`](https://github.com/VCL3D/SingleShotCuboids/blob/584aec312fb381b0a02acd89dd2e299f3fdc7ec5/ssc/quasi_manhattan_center_of_mass.py#L6) module found in [`./ssc/quasi_manhattan_center_of_mass.py`](https://github.com/VCL3D/SingleShotCuboids/blob/master/ssc/quasi_manhattan_center_of_mass.py) estimates the meridian-aligned top and bottom corners using either:
- the `standard` mode that calculates the default center of mass (CoM), or,
- the `periodic` mode which calculates a boundary aware spherical center of mass.

```py
                module = QuasiManhattanCenterOfMass(mode='periodic')
```

Their differences are depicted in the following figure, where the CoM of a set of _blue_ or _pink_ particles, whoses masses are denoted by their size, is estimated with both methods on an equirectangular grid.
The `standard` method (_white filled particles_) fails to properly localize the CoM as it neglects the image's continuity around the horizontal boundary.
The `periodic` method (_darker filled colored particles_) resolves this issue taking into account the continuous boundary.

<img src=./assets/images/boundary_scom2.png width=400/>

The input to the module's `forward` function is:

- a `[W x H]` grid `G` with coordinates normalized to `[-1, 1]`, and,
- the predicted heatmap `H`.

```py
                       corners = scom.forward(grid, gaussian)
```

An example with randomly allocated points, their geodesic gaussian reconstruction and the corresponding localisations using a normalized grid can be seen by running:

```bash
    python ssc/quasi_manhattan_center_of_mass.py '{mode: standard|periodic}'
```

## Cuboid Fitting
The [`CuboidFitting`](https://github.com/VCL3D/SingleShotCuboids/blob/584aec312fb381b0a02acd89dd2e299f3fdc7ec5/ssc/cuboid_fitting.py#L6) module found in [`./ssc/cuboid_fitting.py`](https://github.com/VCL3D/SingleShotCuboids/blob/master/ssc/cuboid_fitting.py) fits a cuboid into `8` estimated corner locations as described in the paper and depicted in the following figure.

```py
                      head = CuboidFitting(mode='joint')
```

![Cuboid Fitting](./assets/images/homography.png "Cuboid Fitting")

A set of examples can be run using:

```bash
      python ssc/cuboid_fitting.py '{test: [1-7]]} {mode: floor|ceil|avg|joint}'
```

where one of `7` test cases can be selected and one of the available modes:

- `floor` for using the floor as a fixed height plane, 
- `ceil`  for using the ceiling as a fixed height plane,
- `avg` for using both and averaging their projected coordinates, and,
- `joint` for fusing the floor view projected floor and ceiling coordinates.

The original coordinates will be colored blue, while the cuboid fitted coordinates will be colored green.

Examples on the different test sets follow, with the images on the left being the predicted coordinates floor plan view, and the images on the right those after cuboid fitting:

### Sun360

<img width=16% src="./assets/images/sun360_1_pred.png"><img width=16% src="./assets/images/sun360_1_cuboid.png"><img width=16% src="./assets/images/sun360_2_pred.png"><img width=16% src="./assets/images/sun360_2_cuboid.png"><img width=16% src="./assets/images/sun360_4_pred.png"><img width=16% src="./assets/images/sun360_4_cuboid.png">

<!--
<img width=45% src="./assets/images/sun360_3_pred.png">
<img width=45% src="./assets/images/sun360_3_cuboid.png">
-->

### Stanford2D3D

<!--
<img width=45% src="./assets/images/s2d3d_1_pred.png">
<img width=45% src="./assets/images/s2d3d_1_cuboid.png">
-->

<img width=16% src="./assets/images/s2d3d_2_pred.png"><img width=16% src="./assets/images/s2d3d_2_cuboid.png"><img width=16% src="./assets/images/s2d3d_3_pred.png"><img width=16% src="./assets/images/s2d3d_3_cuboid.png"><img width=16% src="./assets/images/s2d3d_4_pred.png"><img width=16% src="./assets/images/s2d3d_4_cuboid.png">

### Structure3D

<img width=16% src="./assets/images/s3d_1_pred.png"><img width=16% src="./assets/images/s3d_1_cuboid.png"><img width=16% src="./assets/images/s3d_2_pred.png"><img width=16% src="./assets/images/s3d_2_cuboid.png"><img width=16% src="./assets/images/s3d_3_pred.png"><img width=16% src="./assets/images/s3d_3_cuboid.png">

<!--
<img width=45% src="./assets/images/s3d_4_pred.png">
<img width=45% src="./assets/images/s3d_4_cuboid.png">
-->

### Kujiale

<img width=16% src="./assets/images/kuj_1_pred.png"><img width=16% src="./assets/images/kuj_1_cuboid.png"><img width=16% src="./assets/images/kuj_3_pred.png"><img width=16% src="./assets/images/kuj_3_cuboid.png"><img width=16% src="./assets/images/kuj_4_pred.png"><img width=16% src="./assets/images/kuj_4_cuboid.png">

<!--
<img width=45% src="./assets/images/kuj_2_pred.png">
<img width=45% src="./assets/images/kuj_2_cuboid.png">
-->



## Spherically Padded Convolution

The [`SphericalConv2d`](https://github.com/VCL3D/SingleShotCuboids/blob/584aec312fb381b0a02acd89dd2e299f3fdc7ec5/ssc/spherically_padded_conv.py#L44) module in [`./ssc/spherically_padded_conv.py`](https://github.com/VCL3D/SingleShotCuboids/blob/master/ssc/spherically_padded_conv.py) applies the padding depicted below that adapts traditional convs to the equirectangular domain by replication padding at the singularities/poles and circular padding around the horizontal boundary.

<img src=./assets/images/sconv.png width=400/>

## Citation
If you used or found this code and/or models useful, please cite the following:
```yaml
@arcticle{zioulis2021singleshot,
  author       = "Zioulis, Nikolaos and Alvarez, Federico and Zarpalas, Dimitris and Daras, Petros",
  title        = "Single-Shot Cuboids: Geodesics-based End-to-end Manhattan Aligned Layout Estimation from Spherical Panoramas",
  archivePrefix = {arXiv},  
  month        = "February",
  year         = "2021"
}
```

## Acknowledgements

<img src=./assets/images/atlantis_logo.png width=300>

This project has received funding from the European Union’s Horizon 2020 research and innovation programme [__ATLANTIS__](http://atlantis-ar.eu/) under grant agreement No 951900.