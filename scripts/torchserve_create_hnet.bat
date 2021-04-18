torch-model-archiver --model-name hnet --version 1.0 ^
 -f --model-file ../hnet/model.py ^
 --serialized-file ../ckpt/hnet.pth ^
 --handler ../hnet_handler.py --export-path ../mar ^
 --extra-files ../mesh_handler.py,../boundary_handler.py
pause