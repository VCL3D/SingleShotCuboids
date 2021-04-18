torch-model-archiver --model-name ssc_syn --version 1.0 ^
 -f --model-file ../ssc/model.py ^
 --serialized-file ../ckpt/ssc_syn.pth ^
 --handler ../ssc_handler.py --export-path ../mar ^
 --extra-files ../mesh_handler.py,../boundary_handler.py,^
../ssc/modules/bottleneck.py,../ssc/modules/conv2d.py,^
../ssc/modules/hourglass.py,../ssc/modules/maxpool2d_aa.py,^
../ssc/modules/upsample2d.py,../ssc/cuboid_fitting.py,^
../ssc/quasi_manhattan_center_of_mass.py,^
../ssc/spherical_grid.py
pause