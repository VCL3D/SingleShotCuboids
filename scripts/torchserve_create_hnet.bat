torch-model-archiver --model-name hnet --version 1.0 ^
 -f --model-file ../hnet/model.py ^
 --serialized-file ../ckpts/hnet.pth ^
 --handler ../hnet_handler.py --export-path ../mars ^
 --extra-files ../mesh_handler.py,../boundary_handler.py
pause